"""
Test for generate_and_score function to ensure it correctly extracts output grids.

This test mocks the model generation to verify the extraction logic works correctly.
"""
import sys
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arc.grids.core import from_list
from arc.serialize import pack_example, serialize_grid, BOS, SEP, EOS
from arc.eval.solve_task import generate_and_score

class MockModel:
    """Mock model that returns a pre-defined sequence."""
    
    def __init__(self, output_sequence):
        self.output_sequence = output_sequence
        self.eval_called = False
        self.current_pos = 0
    
    def eval(self):
        self.eval_called = True
        return self
    
    def __call__(self, input_ids):
        """Return logits that would produce our desired output sequence."""
        batch_size, seq_len = input_ids.shape
        vocab_size = 84  # From constants
        
        # Create logits that will produce the next token from our sequence
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        
        # Set the next token position to have highest logit
        if seq_len < len(self.output_sequence):
            next_token = self.output_sequence[seq_len]
            logits[0, -1, next_token] = 100.0  # High value so argmax picks it
        
        # Return as tuple (the greedy_generate expects: logits, = model(cur))
        return (logits,)
    
    def parameters(self):
        """Return a dummy parameter for device detection."""
        return iter([torch.zeros(1, device='cpu')])

def test_generate_and_score_extraction():
    """Test that generate_and_score correctly extracts the output grid."""
    print("\n" + "=" * 60)
    print("Testing generate_and_score Extraction Logic")
    print("=" * 60)
    
    # Create test grids
    x = from_list([[1, 2], [3, 4]])
    y = from_list([[5, 6], [7, 8]])
    
    print(f"\nInput grid:\n{x.a}")
    print(f"\nExpected output grid:\n{y.a}")
    
    # Create the expected packed sequence
    packed = pack_example(x, y, mode="row")
    print(f"\nPacked sequence length: {len(packed)}")
    print(f"Packed sequence: {packed}")
    
    # Mock the greedy_generate function to return our packed sequence
    import arc.models.infer as infer_module
    import arc.eval.scorer as scorer_module
    
    original_greedy_generate = infer_module.greedy_generate
    original_mean_logp = scorer_module.mean_logp_output
    
    def mock_greedy_generate(model, input_ids, max_new_tokens, eos_id):
        """Mock that returns the full packed sequence."""
        # Return the packed sequence as if the model generated it
        return torch.tensor([packed], dtype=torch.long)
    
    def mock_mean_logp(model, output_ids, sep_token_id):
        """Mock scorer that returns a dummy score."""
        return -0.5  # Dummy score
    
    # Temporarily replace functions
    infer_module.greedy_generate = mock_greedy_generate
    scorer_module.mean_logp_output = mock_mean_logp
    
    try:
        # Create a mock model (doesn't need to actually work since we're mocking generation)
        mock_model = MockModel(packed)
        
        # Call generate_and_score
        g_pred, score = generate_and_score(
            mock_model, x, y, 
            device="cpu", 
            mode="row", 
            max_new=512
        )
        
        print(f"\nExtracted grid:\n{g_pred}")
        print(f"Score: {score:.4f}")
        
        # Verify the extracted grid matches the expected output
        assert np.array_equal(g_pred, y.a), \
            f"Extracted grid doesn't match!\nExpected:\n{y.a}\nGot:\n{g_pred}"
        
        print("\n✅ Successfully extracted output grid from packed sequence!")
        
    finally:
        # Restore original functions
        infer_module.greedy_generate = original_greedy_generate
        scorer_module.mean_logp_output = original_mean_logp

def test_generate_and_score_with_different_sizes():
    """Test with different grid sizes to ensure robustness."""
    print("\n" + "=" * 60)
    print("Testing generate_and_score with Different Grid Sizes")
    print("=" * 60)
    
    test_cases = [
        ([[1]], [[2]]),  # 1x1
        ([[1, 2, 3]], [[4, 5, 6]]),  # 1x3
        ([[1], [2], [3]], [[4], [5], [6]]),  # 3x1
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]),  # 3x3
    ]
    
    import arc.models.infer as infer_module
    import arc.eval.scorer as scorer_module
    
    original_greedy_generate = infer_module.greedy_generate
    original_mean_logp = scorer_module.mean_logp_output
    
    # Store the current packed sequence for the mock
    current_packed = []
    
    def mock_greedy_generate(model, input_ids, max_new_tokens, eos_id):
        """Mock that returns the current packed sequence."""
        return torch.tensor([current_packed], dtype=torch.long)
    
    def mock_mean_logp(model, output_ids, sep_token_id):
        """Mock scorer that returns a dummy score."""
        return -0.5
    
    infer_module.greedy_generate = mock_greedy_generate
    scorer_module.mean_logp_output = mock_mean_logp
    
    try:
        for i, (x_list, y_list) in enumerate(test_cases, 1):
            x = from_list(x_list)
            y = from_list(y_list)
            
            print(f"\nTest case {i}: {x.shape} grid")
            print(f"Input:\n{x.a}")
            print(f"Expected output:\n{y.a}")
            
            # Create packed sequence
            packed = pack_example(x, y, mode="row")
            current_packed = packed
            
            # Create mock model
            mock_model = MockModel(packed)
            
            # Test extraction
            g_pred, score = generate_and_score(
                mock_model, x, y,
                device="cpu",
                mode="row",
                max_new=512
            )
            
            print(f"Extracted:\n{g_pred}")
            
            # Verify
            assert np.array_equal(g_pred, y.a), \
                f"Test case {i} failed! Extracted grid doesn't match expected."
            
            print(f"✅ Test case {i} passed!")
            
    finally:
        infer_module.greedy_generate = original_greedy_generate
        scorer_module.mean_logp_output = original_mean_logp
    
    print("\n✅ All grid size tests passed!")

def test_generate_and_score_error_handling():
    """Test error handling for malformed sequences."""
    print("\n" + "=" * 60)
    print("Testing generate_and_score Error Handling")
    print("=" * 60)
    
    x = from_list([[1, 2], [3, 4]])
    y = from_list([[5, 6], [7, 8]])
    
    import arc.models.infer as infer_module
    import arc.eval.scorer as scorer_module
    
    original_greedy_generate = infer_module.greedy_generate
    original_mean_logp = scorer_module.mean_logp_output
    
    def mock_mean_logp(model, output_ids, sep_token_id):
        return -0.5
    
    scorer_module.mean_logp_output = mock_mean_logp
    
    # Test 1: No EOS token
    print("\nTest 1: Sequence with no EOS token")
    def mock_no_eos(model, input_ids, max_new_tokens, eos_id):
        return torch.tensor([[BOS, 5, 35, SEP]], dtype=torch.long)
    
    infer_module.greedy_generate = mock_no_eos
    mock_model = MockModel([])
    
    try:
        g_pred, score = generate_and_score(mock_model, x, y, device="cpu", mode="row")
        print("❌ Should have raised ValueError for missing EOS")
        assert False, "Should have raised ValueError"
    except (ValueError, IndexError) as e:
        print(f"✅ Correctly raised error: {type(e).__name__}: {e}")
    
    # Test 2: No SEP after EOS
    print("\nTest 2: Sequence with no SEP after first EOS")
    def mock_no_sep_after_eos(model, input_ids, max_new_tokens, eos_id):
        # Input grid ending with EOS, but no SEP and output
        return torch.tensor([[BOS, 5, 35, SEP, 65, 66, SEP, 75, 76, EOS]], dtype=torch.long)
    
    infer_module.greedy_generate = mock_no_sep_after_eos
    
    try:
        g_pred, score = generate_and_score(mock_model, x, y, device="cpu", mode="row")
        print("❌ Should have raised ValueError for missing SEP after EOS")
        assert False, "Should have raised ValueError"
    except (ValueError, IndexError) as e:
        print(f"✅ Correctly raised error: {type(e).__name__}: {e}")
    
    # Restore
    infer_module.greedy_generate = original_greedy_generate
    scorer_module.mean_logp_output = original_mean_logp
    
    print("\n✅ Error handling tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing generate_and_score Function")
    print("=" * 60)
    
    try:
        test_generate_and_score_extraction()
        test_generate_and_score_with_different_sizes()
        test_generate_and_score_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL generate_and_score TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

