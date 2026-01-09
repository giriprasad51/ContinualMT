from typing import Dict, List, Optional
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from scipy.stats import norm
from math import sqrt, pi
from fairseq.modules import GradMultiply

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from itertools import combinations, product
import torch.distributed as dist

aux_info = {}
class HATLayer(nn.Module):
    """HAT Layer block.

    Args:
        cfg (FairseqDataclass): HAT config
    """

    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.task_embedding = negative_embedding(cfg.hat.task_num, embed_dim)
        self.activation_gate = nn.Sigmoid()
        self.dummy_param = nn.Parameter(torch.empty(0))
              

    def mask(self, temperature=None, task_id=None):
        """Compute the mask for HAT layer.
        The mask is a sigmoid function of task embedding.

        Returns:
            Tensor: mask tensor with shape `(1, embed_dim)`
        """

        device = self.dummy_param.device
        
        #return torch.ones(self.embed_dim).data.detach().to(device)


        if task_id is None:
            task_id = self.cfg.hat.task_id
        if temperature is None:
            temperature = self.cfg.hat.temperature

        embedding = self.task_embedding(torch.LongTensor([task_id]).to(device))

        if self.is_training:
            mask = self.activation_gate(temperature*embedding)
        else:
            mask = self.activation_gate(self.cfg.hat.temperature_max*embedding)

        return mask


    def get_previous_task_mask(self):
        """Compute the mask for all previous tasks.

        Returns:
            Tensor: mask tensor with shape `(1, embed_dim)`
        """
        device = self.dummy_param.device   

        previous_mask = torch.zeros(self.embed_dim).to(device)
        
        # if task_id is not 0, return a mask that combines all previous tasks' mask
        for i in range(self.cfg.hat.task_id):
            mask = self.mask(temperature=self.cfg.hat.temperature_max, task_id=i)
            previous_mask = torch.max(previous_mask, mask)

        previous_mask = (previous_mask > 0.5).to(torch.float)
        # print(previous_mask)

        return previous_mask.data.detach().to(device)

    def forward(self, x, task_id=None):
        """Compute the HAT layer.

        Args:
            x (Tensor): input tensor with shape `(seq_len, batch, embed_dim)`

        Returns:
            Tensor: output tensor with shape `(seq_len, batch, embed_dim)`
        """
        # (seq_len, batch, embed_dim)
        mask = self.mask(temperature=self.cfg.hat.temperature, task_id=task_id)
        #print(mask)
        x = x * mask
        # x = GradMultiply.apply(x, mask)

        return x


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    for idx in range(num_embeddings):
        mean = norm.ppf(1 / (num_embeddings - idx) * 0.9999)
        nn.init.normal_(m.weight.data[idx], mean=mean, std=1)
        # normalize embedding
        m.weight.data[idx] = m.weight.data[idx] / m.weight.data[idx].norm(2, -1, keepdim=True)
        
    return m

def negative_embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    # generate normal distribution, make all embeddings negative
    for idx in range(num_embeddings):
        m.weight.data[idx] = torch.abs(m.weight.data[idx]) * -1
        
    return m



class SVDMaskLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = None,
        device=None,
    ):
        
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.layer = linear
        self.device = device or linear.weight.device
        self.compute_svd()

    

    def compute_svd(self):
        """Compute SVD of the weight matrix"""
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(self.layer.weight, full_matrices=True)
            self.U = U.detach()
            self.S = S.detach()
            self.Vt = Vt.detach()
            # print(S)
            # print(U.shape, S.shape, Vt.shape)

    def scale_sorted_chunk_inplace(self, S, alpha=None, beta=None):
    
        ind = torch.arange(S.shape[0], device=S.device)
        S_scaled = S.clone().to(alpha.device)
        # print(S_scaled.shape)
        # Chunk indices
        sizes = [len(ind) // len(alpha) + (1 if i < len(ind) % len(alpha) else 0) for i in range(len(alpha)) ]
        chunks = torch.split(ind, sizes)
        if beta==None:
            for i in range(len(alpha)):
                S_scaled[chunks[i]] =  S_scaled[chunks[i]]*alpha[i] 
        else:
            for i in range(len(alpha)):
                S_scaled[chunks[i]] =  S_scaled[chunks[i]]*alpha[i] +beta[i]
        return S_scaled

    

    def reconstructed_weight(self, alpha=None, beta=None):
        if alpha==None or (alpha==1).all().item():
            return self.layer.weight
        
        """Reconstruct masked weight"""
        # if S==None:
        S = self.scale_sorted_chunk_inplace(self.S, alpha=alpha, beta=beta)
        m,n = self.layer.weight.shape
        Sigma = torch.zeros((m, n), device=self.layer.weight.device, dtype=self.layer.weight.dtype)
        Sigma[:min(m, n), :min(m, n)] = torch.diag(S)
        self.U = self.U.to(Sigma.device)
        self.Vt = self.Vt.to(Sigma.device)
        # print(self.U.device , Sigma.device , self.Vt.device)
        return self.U @ Sigma @ self.Vt

    def forward(self, x, alpha=None,beta=None):
        # print(self.S.shape, alpha.shape)
        
        W_rec = self.reconstructed_weight( alpha=alpha,beta=beta)
       
        return  F.linear(x, W_rec, self.layer.bias.detach())


class SparsityAwareRouter(nn.Module):
    """
    Sparsity-aware routing mechanism that selects experts based on 
    predicted activation sparsity rather than just performance.
    
    Key innovation: Routes to experts with highest expected sparsity
    using Gaussian approximation of pre-activations.
    """
    def __init__(self, in_features: int, n_experts: int, top_k: int, alpha, beta):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.in_features = in_features
        self.alpha = alpha
        self.beta = beta
        
    def compute_expert_statistics(self, x: torch.Tensor, W=None, bias=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute μ_h and σ_h for each expert without full forward pass.
        
        Uses column-wise statistics of encoder weights to estimate
        the mean and std of pre-activations.
        
        Args:
            expert: ReLUExpert module
            x: Input [batch_size, seq_len, in_features]
        Returns:
            mu_h: Mean of pre-activations [batch_size, seq_len]
            sigma_h: Std of pre-activations [batch_size, seq_len]
        """
        
        # Column-wise mean and variance of encoder weights
        mu_weights = W.mean(dim=0)  # [in_features]
        var_weights = W.var(dim=0, unbiased=False)  # [in_features]
        
        # Compute μ_h = μ^T x + b_mean
        # print(x.shape, mu_weights.shape,bias.mean().shape)
        mu_h = torch.matmul(x, mu_weights) + bias.mean()  # [batch, seq_len]
        
        # Compute σ_h = sqrt(σ^T (x^2))
        x_squared = x ** 2
        sigma_h_squared = torch.matmul(x_squared, var_weights)
        sigma_h = torch.sqrt(sigma_h_squared + 1e-8)  # [batch, seq_len]
        
        return mu_h, sigma_h
    
    def forward(self, x: torch.Tensor, expert: nn.ModuleList, 
                is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to top-k sparsest experts.
        
        Args:
            x: Input [batch_size, seq_len, in_features]
            experts: List of ReLUExpert modules
            training: Whether in training mode
        Returns:
            router_weights: Gating weights [batch_size, seq_len, n_experts]
            router_logits: Raw routing scores [batch_size, seq_len, n_experts]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute sparsity scores for each expert
        sparsity_scores = []
        for alpha, beta in zip(self.alpha, self.beta):
            W = expert.reconstructed_weight(alpha=alpha,beta=beta)
            mu_h, sigma_h = self.compute_expert_statistics(x,W,expert.layer.bias)
            
            # Approximate Φ(μ_h / σ_h) using erf
            # Lower score = sparser activation (fewer positive values)
            z_score = mu_h / (math.sqrt(2) * sigma_h)
            phi_score = torch.erf(z_score)  # Approximates CDF
            
            # We want to minimize Φ, so use negative
            sparsity_scores.append(phi_score)

        # print(sparsity_scores)
        
        # Stack scores: [batch_size, seq_len, n_experts]
        router_logits = torch.stack(sparsity_scores, dim=-1)
        
        # Apply top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Apply softmax to top-k
        router_weights = torch.zeros_like(router_logits)
        router_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))
        # print(router_weights.shape, router_logits)
        
        return router_weights, router_logits


class MoEXLayer(nn.Module):
    """
    MoE-X layer combining ReLU experts with sparsity-aware routing.
    
    This layer acts as a wide, sparse MLP by:
    1. Using multiple ReLU experts for sparse activations
    2. Routing via sparsity prediction
    3. Combining expert outputs with learned weights
    """
    def __init__(self, expert: nn.ModuleList, top_k: int,  device=None):
        super().__init__()


        self.expert = expert
        self.device = device or self.expert.layer.weight.device
        self.top_k = top_k
        self.alpha = nn.ParameterList([nn.Parameter(torch.ones(min(self.expert.layer.weight.shape)).to(self.device))] )
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(min(self.expert.layer.weight.shape)).to(self.device))] )
        
        self.n_experts = len(self.alpha)
        self.is_training = True

        self.in_features = self.expert.layer.in_features
        # global self.load_balancing
        self.load_balancing = {}
        self.router = SparsityAwareRouter(
            in_features=self.in_features,
            n_experts=self.n_experts,
            top_k=top_k,
            alpha=self.alpha,
            beta=self.beta
        )
    def add_alpha(self,n=1, n_params=8):
        self.alpha += nn.ParameterList([nn.Parameter(nn.Linear(1,min(self.expert.layer.weight.shape) ).weight.squeeze().to(self.device)) for i in range(n)] ) 
        self.beta += nn.ParameterList([nn.Parameter(nn.Linear(1,min(self.expert.layer.weight.shape) ).weight.squeeze().to(self.device)) for i in range(n)] ) 
        
    def compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage balanced expert usage.
        
        Args:
            router_probs: [batch_size, seq_len, n_experts]
        Returns:
            load_balance_loss: Scalar tensor
        """
        # Average probability of routing to each expert
        expert_usage = router_probs.mean(dim=[0, 1])  # [n_experts]
        
        # We want uniform distribution (1/n_experts for each)
        target = 1.0 / self.n_experts
        
        # MSE loss
        load_balance_loss = ((expert_usage - target) ** 2).sum()
        
        return load_balance_loss
    
    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Args:
            x: [batch, seq, in_features]
        Returns:
            output: [batch, seq, in_features]
            aux_info: routing alphagnostics
        """
        B, S, D = x.shape
        # print(self.is_training)
        # print(hugfghj)
        if not self.is_training:
            # print("---------MoE-X in eval mode-------------", self.is_training)
            # ---------- Routing ----------
            router_weights, router_logits = self.router(
                x, self.expert, self.is_training
            )
    
            # top-k per token
            topk_vals, topk_idx = torch.topk(
                router_logits, self.top_k, dim=-1
            )
            topk_weights = F.softmax(topk_vals, dim=-1)

            # ---------- Prepare flattened views ----------
            x_flat = x.view(-1, D)                       # [B*S, D]
            output = torch.zeros(
                *x.shape[:-1],
                self.expert.layer.weight.shape[0],
                device=x.device,
                dtype=x.dtype
            )
            output_flat = output.view(-1, self.expert.layer.weight.shape[0])
    
            topk_idx_flat = topk_idx.view(-1, self.top_k)
            topk_w_flat = topk_weights.view(-1, self.top_k)
    
            expert_activations = [None] * self.n_experts
            active_experts = []
            # print(router_logits.shape,topk_vals.shape)
            # print(topk_idx_flat)
            # ---------- Token → Expert Dispatch ----------
            global aux_info
            self.load_balancing =  aux_info.get("load_balancing", {})
            # print(aux_info.get("load_balancing", {}))
            for expert_id in range(self.n_experts):
                mask = topk_idx_flat == expert_id        # [B*S, K]
                # print("----------",expert_id,mask)
                if not mask.any():
                    continue
    
                active_experts.append(expert_id)
    
                token_ids, k_ids = mask.nonzero(as_tuple=True)
    
                tokens = x_flat[token_ids]               # [N, D]
                weights = topk_w_flat[token_ids, k_ids].unsqueeze(-1)
                
                # print(expert_id,  token_ids)
                if expert_id  in self.load_balancing:
                    self.load_balancing[expert_id] += len(token_ids)
                else:
                    self.load_balancing[expert_id] = len(token_ids)
                # expert forward ONLY on routed tokens
                expert_out = self.expert(tokens.unsqueeze(1), self.alpha[expert_id])
                expert_out = expert_out.squeeze(1)
                # print(weights.shape, expert_out.shape, output.shape, output_flat.shape)
                # weighted accumulation
                output_flat.index_add_(
                    0, token_ids, expert_out * weights
                )
    
            # print(self.load_balancing)
    
            # ---------- Auxiliary Info ----------
            # print("-----------MOE-X------------------")
            
            if return_aux:
                load_balance_loss = self.compute_load_balance_loss(router_weights)
    
               
    
                aux_info = {
                    "router_weights": router_weights,
                    "router_logits": router_logits,
                    "load_balance_loss": aux_info.get("load_balance_loss", 0.0)+load_balance_loss,
                    "active_experts": torch.tensor(active_experts, device=x.device),
                    "load_balancing": self.load_balancing,
                    # "load_balancing": {k: load_balancing.get(k, 0) + aux_info.get("load_balancing", {}).get(k, 0) for k in load_balancing.keys() | aux_info.get("load_balancing", {}).keys()}
                }
    
            return output#, aux_info

        else:
            # print("---------MoE-X in train mode-------------", self.is_training)
            task_id = int(os.environ.get("alpha_ID", None))
            
            return self.expert(x, self.alpha[task_id], self.beta[task_id])


if __name__ == "__main__":
    from approaches.models.hat.hat_transformer_ffn_config import HATTransformerConfig

    cfg = HATTransformerConfig()
    cfg.hat.task_num = 5
    cfg.hat.temperature = 0.1
    cfg.hat.temperature_max = 10
    cfg.hat.thres_cosh = 2
    cfg.hat.thres_emb = 0.5
    cfg.hat.task_id = 0
    embed_dim = 16
    hat_layer = HATLayer(cfg, embed_dim).to("cuda")
    x = torch.rand(32, 16, embed_dim).to("cuda")
    mask = hat_layer.mask()

    print(mask)
    loss = torch.sum(mask)
    print(loss)

    loss.backward()

    for name, param in hat_layer.named_parameters():
        print(name, param.grad)
