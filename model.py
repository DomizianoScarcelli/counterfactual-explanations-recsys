import torch
import torch.nn as nn
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


class MockRecommenderSystem(RecommenderSystem):
    def __init__(self, num_items: int, top_k: int):
        """
        Args:
            num_items: Total number of items available for recommendation
            top_k: Number of top items to recommend
        """
        super(MockRecommenderSystem, self).__init__()
        self.num_items = num_items
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: A tensor representing user interaction history, shape (batch_size, num_items)
        
        Returns:
            A tensor of size (batch_size, top_k) with the indices of the top-k recommendations.
        """
        # Mock logic: Recommend the first top_k items for every user
        # In a real system, this might involve more complex logic like matrix factorization, attention models, etc.
        
        # For simplicity, here we recommend items 0, 1, ..., top_k-1 for every user
        batch_size = x.size(0)
        top_k_items = torch.arange(self.top_k).expand(batch_size, -1)  # Shape: (batch_size, top_k)
        
        return top_k_items


# Example usage:
if __name__ == "__main__":
    num_items = 100  # Mock number of items
    top_k = 5  # We want the top 5 recommendations

    recommender = MockRecommenderSystem(num_items=num_items, top_k=top_k)
    
    # Mock input: A batch of 2 users, with interaction history (we'll leave the actual values unused)
    user_history = torch.rand((2, num_items))  # Shape: (batch_size=2, num_items=100)
    
    recommendations = recommender(user_history)
    print("Top-k Recommendations:\n", recommendations)
