from torch.utils.data import Dataset
import torch

class MockSequentialDataset(Dataset):
    def __init__(self, num_users: int, num_items: int, max_seq_len: int):
        """
        Args:
            num_users: The number of users in the dataset.
            num_items: The total number of items available.
            max_seq_len: Maximum length of interaction sequence per user. If a user has fewer interactions, the sequence is padded.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.max_seq_len = max_seq_len

        # Generate random sequential interaction history for each user
        self.data = []
        for _ in range(num_users):
            # Random sequence length for this user (between 1 and max_seq_len)
            seq_len = torch.randint(1, max_seq_len + 1, (1,)).item()

            # Random sequence of item IDs (between 0 and num_items - 1)
            interaction_sequence = torch.randint(0, num_items, (seq_len,))
            
            # Store the interaction sequence for this user
            self.data.append(interaction_sequence)

    def __len__(self):
        # Length is the number of users
        return self.num_users

    def __getitem__(self, index):
        """
        Return the interaction sequence for the user at the given index.
        
        Args:
            index: User index.
        
        Returns:
            A tuple (padded_sequence, target_item, seq_len):
                - padded_sequence: Tensor of shape (max_seq_len,) with the user's interaction history, padded with 0.
                - target_item: The next item the user interacted with (for next-item prediction), or 0 if there is no next item.
                - seq_len: The actual length of the interaction sequence (before padding).
        """
        interaction_sequence = self.data[index]
        seq_len = len(interaction_sequence)

        # Next item prediction: target is the last item in the sequence
        target_item = interaction_sequence[-1] if seq_len > 1 else 0

        # Pad the sequence to max_seq_len with zeros
        padded_sequence = torch.zeros(self.max_seq_len, dtype=torch.long)
        padded_sequence[:seq_len] = interaction_sequence

        return padded_sequence, target_item, seq_len


# Example usage:
if __name__ == "__main__":
    num_users = 10  # Mock number of users
    num_items = 100  # Mock number of items
    max_seq_len = 20  # Maximum sequence length

    dataset = MockSequentialDataset(num_users=num_users, num_items=num_items, max_seq_len=max_seq_len)

    # Fetch the interaction history of the first user
    user_0_data = dataset[0]
    padded_sequence, target_item, seq_len = user_0_data

    print("User 0 Interaction History (Padded Sequence):\n", padded_sequence)
    print("User 0 Target Item (for next-item prediction):", target_item)
    print("User 0 Actual Sequence Length:", seq_len)

    # Print the total number of users in the dataset
    print("Number of users in dataset:", len(dataset))
