from evaluations.evaluator import CustomEvaluator
import sys

checkpoint_path = "/Users/girish/Documents/Mech Interp/Mine/checkpoints/20260113170811/sft/final_model.pt"
device = "mps"

try:
    print("Initializing Evaluator...")
    evaluator = CustomEvaluator(checkpoint_path=checkpoint_path, device=device, batch_size=4)
    
    print("Running evaluation...")
    tasks = ["hellaswag"]
    results = evaluator.evaluate(tasks, limit=5)
    
    print("\nFINAL RESULTS STRUCTURE:")
    import json
    # Use default=str to handle non-serializable objects if any
    print(json.dumps(results, indent=2, default=str))

except Exception as e:
    import traceback
    traceback.print_exc()
