import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class RecommenderSystem(ABC, nn.Module):
    def __init__(self):
        super(RecommenderSystem, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: the history of interactions, represented as a torch.Tensor of size (batch_size, num_items)
        Returns:
            A torch.Tensor representing the top-k recommendations given the input x.
        """
        pass


class MockRecommenderSystem(nn.Module):
    def __init__(self, num_items: int, top_k: int, embedding_dim: int):
        """
        Args:
            num_items: Total number of items available for recommendation
            top_k: Number of top items to recommend
            embedding_dim: The dimensionality of the embedding space
        """
        super(MockRecommenderSystem, self).__init__()
        self.num_items = num_items
        self.top_k = top_k
        self.embedding_dim = embedding_dim
        
        # Randomly initialized item embeddings in a lower-dimensional space
        self.item_embeddings = nn.Parameter(torch.randn(num_items, embedding_dim))
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: A tensor representing user interaction history, shape (batch_size, num_features)
            num_features can be different from num_items
        
        Returns:
            A tensor of size (batch_size, top_k) with the indices of the top-k recommendations.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size, num_features = x.size()

        # Embedding padding or truncation as needed
        if num_features < self.num_items:
            padding_size = self.num_items - num_features
            x = F.pad(x, (0, padding_size), "constant", 0)
        elif num_features > self.num_items:
            x = x[:, :self.num_items]
        
        # Project user interaction history to the embedding space
        x_embeddings = torch.matmul(x, self.item_embeddings)  # Shape: (batch_size, embedding_dim)

        # Compute scores for all items (dot product in embedding space)
        scores = torch.matmul(x_embeddings, self.item_embeddings.T)  # Shape: (batch_size, num_items)
        
        # Get the indices of the top-k items per user
        _, top_k_items = torch.topk(scores, self.top_k, dim=1)  # Shape: (batch_size, top_k)
        
        return top_k_items

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(42)
    num_items = 10_000 # Mock number of items
    top_k = 5  # We want the top 5 recommendations

    recommender = MockRecommenderSystem(num_items=num_items, top_k=top_k, embedding_dim=64)
    
    # Mock input: A batch of 2 users, with interaction history (we'll leave the actual values unused)
    user_history = torch.randint(high=num_items, size=(2, 1500)).float()  # Shape: (batch_size=2, num_items=100)
    
    recommendations = recommender(user_history)
    print("Top-k Recommendations:\n", recommendations)
