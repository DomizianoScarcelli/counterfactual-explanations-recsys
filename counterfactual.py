from model import RecommenderSystem, MockRecommenderSystem
from torch.utils.data import Dataset
import torch
from typing import Tuple, List
import heapq
from tqdm import tqdm

class Explainer:
    def __init__(self, 
                 recommender: RecommenderSystem):
        self.recommender = recommender

    def apply_swap(self, sequence: torch.Tensor, idx_src: int, idx_dst: int) -> Tuple[torch.Tensor, float]:
        """
        Swaps two items in the sequence.
        Args:
            sequence: torch.Tensor that represents the user history.
            idx_src: the index of the item that will be swapped to idx_dest
            idx_dest: the index of the items that will be swapped to idx_src
        Returns:
            A tuple where the first element is the new sequence edited with the
            swap , and the second element is the cost of the edit, indicated as
            the Euclidean distance between the original and the edited
            sequence.
        """
        # Clone the original sequence to avoid modifying it in place
        new_sequence = sequence.clone()

        # Perform the swap
        new_sequence[idx_src], new_sequence[idx_dst] = new_sequence[idx_dst], new_sequence[idx_src]
        
        cost = torch.dist(new_sequence.float(), sequence.float(), p=2).item()
        return new_sequence, cost

    def apply_replacement(self, sequence: torch.Tensor, idx: int, new_value: int) -> Tuple[torch.Tensor, float]:
        """
        Replace an item in the sequence with another item
        Args:
            sequence: torch.Tensor that represents the user history.
            idx: the index of the item that will be replaced 
            new_value: the new value of the replaced item 
        Returns:
            A tuple where the first element is the new sequence edited with the
            replacement, and the second element is the cost of the edit,
            indicated as the Euclidean distance between the original and the
            edited sequence.
        """
        # Clone the original sequence to avoid modifying it in place
        new_sequence = sequence.clone()

        # Replace the value at the given index with the new value
        new_sequence[idx] = new_value
        cost = torch.dist(new_sequence.float(), sequence.float(), p=2).item()
        
        return new_sequence, cost

    def apply_deletion(self, sequence: torch.Tensor, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Delete an item from the sequence
        Args:
            sequence: torch.Tensor that represents the user history.
            idx: the index of the item that will be deleted
        Returns:
            A tuple where the first element is the new sequence edited with the
            deletion and the second element is the cost of the edit, indicated
            as the Euclidean distance between the original and the edited
            sequence. Note that the length of the sequence will be the same,
            all the values after the deleted index will be shifted left by one,
            and a 0 will be added to the end as padding.
        """
        # Clone the original sequence to avoid modifying it in place
        new_sequence = sequence.clone()

        # Shift all elements after the deleted index to the left
        if idx < len(new_sequence) - 1:
            new_sequence[idx:-1] = new_sequence[idx+1:].clone() #clone, otherwise error
        
        # Set the last element to 0 (padding)
        new_sequence[-1] = 0
        
        cost = torch.dist(new_sequence.float(), sequence.float(), p=2).item()
        return new_sequence, cost
    
    @torch.no_grad()
    def a_star(self, sequence: torch.Tensor, max_steps: int = 100_000) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform A* search to find counterfactual explanation.
        Args:
            sequence: Original sequence of interactions.
            max_steps: Maximum number of steps before halting the search.
        Returns:
            The best counterfactual sequence and the sequence of edits.
        """
        def heuristic(seq: torch.Tensor) -> float:
            """ Heuristic based on recommendation scores (e.g., distance in scores). """
            rec_original = self.recommender(sequence)[0,0].float()
            rec_current = self.recommender(seq)[0,0].float()
            # return torch.dist(rec_original, rec_current, p=2).item()
            return abs(rec_original - rec_current).item()

        # Priority queue to store (cost, sequence, edit path)
        pq = [(0, sequence, [])]
        visited = set()
        visited.add(tuple(sequence.tolist()))  # Store sequences as tuples for immutability in the visited set
        pbar = tqdm(total=max_steps)

        original_rec = self.recommender(sequence)[0, 0].item()

        while pq and len(pq) < max_steps:
            pbar.update(1)
            cost, current_sequence, path = heapq.heappop(pq)
            current_sequence = torch.tensor(current_sequence)

            current_rec = self.recommender(current_sequence)[0, 0].item()

            # Check if this is a valid counterfactual (change in recommendation)
            if current_rec != original_rec:
                return current_sequence, path

            # Apply transitions: swap, replacement, deletion
            for idx_src in range(len(current_sequence)):
                for idx_dst in range(len(current_sequence)):
                    if idx_src != idx_dst:
                        new_seq, transition_cost = self.apply_swap(current_sequence, idx_src, idx_dst)
                        new_path = path + [('swap', idx_src, idx_dst)]
                        total_cost = cost + transition_cost + heuristic(new_seq)

                        if tuple(new_seq.tolist()) not in visited:
                            visited.add(tuple(new_seq.tolist()))
                            heapq.heappush(pq, (-total_cost, new_seq.tolist(), new_path))
                
                # Try replacements and deletions
                for new_value in range(len(sequence)):  # Assuming items are integers
                    new_seq, transition_cost = self.apply_replacement(current_sequence, idx_src, new_value)
                    new_path = path + [('replace', idx_src, new_value)]
                    total_cost = cost + transition_cost + heuristic(new_seq)

                    if tuple(new_seq.tolist()) not in visited:
                        visited.add(tuple(new_seq.tolist()))
                        heapq.heappush(pq, (-total_cost, new_seq.tolist(), new_path))
                
                new_seq, transition_cost = self.apply_deletion(current_sequence, idx_src)
                new_path = path + [('delete', idx_src)]
                total_cost = cost + transition_cost + heuristic(new_seq)

                if tuple(new_seq.tolist()) not in visited:
                    visited.add(tuple(new_seq.tolist()))
                    heapq.heappush(pq, (-total_cost, new_seq.tolist(), new_path))
        
        return None, []
    
    def explain(self, sequence: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    num_items, top_k = 1_000, 10
    mock_recommender = MockRecommenderSystem(num_items=num_items, top_k=top_k, embedding_dim=64)
    explainer = Explainer(mock_recommender)
    
    history_length = 100
    sequence = torch.randint(high=num_items, size=(history_length,)).float()  # Shape: (batch_size=2, num_items=100)
    counterfactual, edit_path = explainer.a_star(sequence)
    print("Original sequence:", sequence)
    print("Counterfactual sequence:", counterfactual)
    print("Edit path:", edit_path)
