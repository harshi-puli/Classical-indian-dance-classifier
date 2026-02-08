import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# Set style
sns.set_style('whitegrid')

def explore_data(csv_path, img_dir, dataset_name='Train'):
    """Explore the dance dataset"""
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset Analysis")
    print(f"{'='*60}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    # Class distribution
    print(f"\n{'-'*60}")
    print("Class Distribution:")
    print(f"{'-'*60}")
    class_counts = df['target'].value_counts().sort_index()
    print(class_counts)
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='steelblue')
    plt.title(f'{dataset_name} Dataset - Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Dance Form', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nClass distribution plot saved as '{dataset_name.lower()}_class_distribution.png'")
    
    # Check for missing images
    print(f"\n{'-'*60}")
    print("Checking image files...")
    print(f"{'-'*60}")
    missing_images = []
    existing_images = []
    
    for img_name in df['Image']:
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            existing_images.append(img_name)
        else:
            missing_images.append(img_name)
    
    print(f"Total images in CSV: {len(df)}")
    print(f"Existing images: {len(existing_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images are missing!")
        print(f"First 5 missing: {missing_images[:5]}")
    
    # Sample image dimensions
    if existing_images:
        print(f"\n{'-'*60}")
        print("Analyzing image dimensions (first 20 images)...")
        print(f"{'-'*60}")
        dimensions = []
        
        for img_name in existing_images[:20]:
            img_path = os.path.join(img_dir, img_name)
            try:
                img = Image.open(img_path)
                dimensions.append(img.size)
            except Exception as e:
                print(f"Error reading {img_name}: {e}")
        
        if dimensions:
            widths, heights = zip(*dimensions)
            print(f"Width range: {min(widths)} - {max(widths)} pixels")
            print(f"Height range: {min(heights)} - {max(heights)} pixels")
            print(f"Average dimensions: {int(np.mean(widths))} x {int(np.mean(heights))}")
    
    return df, class_counts

def visualize_samples(csv_path, img_dir, n_samples=3):
    """Visualize sample images from each class"""
    
    df = pd.read_csv(csv_path)
    classes = sorted(df['target'].unique())
    
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(15, 4*len(classes)))
    fig.suptitle('Sample Images from Each Dance Form', fontsize=16, fontweight='bold', y=0.995)
    
    for i, dance_class in enumerate(classes):
        # Get samples from this class
        class_samples = df[df['target'] == dance_class].head(n_samples)
        
        for j, (idx, row) in enumerate(class_samples.iterrows()):
            img_path = os.path.join(img_dir, row['Image'])
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    
                    if j == 0:
                        axes[i, j].set_title(f'{dance_class}\n{row["Image"]}', 
                                            fontsize=10, fontweight='bold')
                    else:
                        axes[i, j].set_title(row['Image'], fontsize=9)
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error loading\n{row["Image"]}', 
                                   ha='center', va='center')
                    axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, f'Missing\n{row["Image"]}', 
                               ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("\nSample images visualization saved as 'sample_images.png'")

def main():
    """Main function for exploratory data analysis"""
    
    # Update these paths according to your directory structure
    TRAIN_CSV = 'dataset/train.csv'
    TEST_CSV = 'dataset/validation_split.csv'
    TRAIN_IMG_DIR = 'dataset/train'
    TEST_IMG_DIR = 'dataset/test'
    
    print("\n" + "="*60)
    print("INDIAN CLASSICAL DANCE IMAGE DATASET - EXPLORATORY ANALYSIS")
    print("="*60)
    
    # Explore training data
    train_df, train_counts = explore_data(TRAIN_CSV, TRAIN_IMG_DIR, 'Train')
    
    # Explore test data
    test_df, test_counts = explore_data(TEST_CSV, TEST_IMG_DIR, 'Test')
    
    # Compare distributions
    print(f"\n{'='*60}")
    print("Train vs Test Distribution Comparison")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).fillna(0).astype(int)
    
    print(comparison_df)
    
    # Plot comparison
    comparison_df.plot(kind='bar', figsize=(12, 6), color=['steelblue', 'coral'])
    plt.title('Train vs Test Dataset Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Dance Form', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('train_test_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'train_test_comparison.png'")
    
    # Visualize sample images
    print(f"\n{'='*60}")
    print("Generating Sample Image Visualization...")
    print(f"{'='*60}")
    visualize_samples(TRAIN_CSV, TRAIN_IMG_DIR, n_samples=3)
    
    print(f"\n{'='*60}")
    print("Exploratory Analysis Complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Review the generated visualizations")
    print("2. Check if class distribution is balanced")
    print("3. Verify all images are accessible")
    print("4. Run 'dance_classifier.py' to start training")

if __name__ == '__main__':
    main()