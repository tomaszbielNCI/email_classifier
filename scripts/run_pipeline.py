#!/usr/bin/env python3
"""
Run Email Classification Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.preprocessing.pipeline import EmailClassificationPipeline
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the pipeline"""
    logger.info("Starting Email Classification Pipeline")
    
    try:
        # Initialize pipeline
        pipeline = EmailClassificationPipeline()
        logger.info("Pipeline initialized successfully")
        
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            data_path="data/raw/AppGallery.csv",
            target_columns=["Type2", "Type3", "Type4"],
            text_columns=["Ticket Summary", "Interaction content"]
        )
        
        logger.info("Pipeline completed successfully")
        
        # Print results
        print("\n🎉 **Pipeline Results:**")
        print(f"Best model: {results['modeling_results']['best_model']}")
        print(f"Best score: {results['modeling_results']['best_score']:.4f}")
        
        # Save results
        import json
        with open("results/pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Results saved to results/pipeline_results.json")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
