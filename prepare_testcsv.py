"""
Helper Script: Prepare Test CSV with Labels
============================================

This script helps you create a proper test.csv file with labels.

There are two scenarios:
1. You HAVE labels for test images → Create test.csv with labels
2. You DON'T have labels → Use this for prediction only
"""

import pandas as pd
import os

def create_test_csv_with_labels():
    """
    If you know the labels for your test images, use this function.
    You'll need to manually create a mapping of image -> label
    """
    
    print("="*60)
    print("OPTION 1: Create test.csv WITH labels")
    print("="*60)
    
    # Example: Manually define your test labels
    # Replace this with YOUR actual test image labels
    test_data = {
        'Image': [
            '508.jpg', '246.jpg', '473.jpg', '485.jpg', '128.jpg',
            # Add ALL your test image filenames here
        ],
        'target': [
            'odissi', 'mohiniyattam', 'odissi', 'odissi', 'bharatanatyam',
            # Add the corresponding labels here
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_with_labels.csv', index=False)
    
    print(f"✓ Created test_with_labels.csv with {len(df)} labeled images")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    return df


def split_train_into_train_val():
    """
    If you DON'T have test labels, split your training data into:
    - Training set (80%)
    - Validation/Test set (20%)
    """
    
    print("="*60)
    print("OPTION 2: Split train.csv into train + validation sets")
    print("="*60)
    
    # Load original training data
    train_df = pd.read_csv('train.csv')
    print(f"\nOriginal training set: {len(train_df)} images")
    
    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: 80% train, 20% validation
    split_idx = int(len(train_df) * 0.8)
    
    new_train = train_df[:split_idx]
    validation = train_df[split_idx:]
    
    # Save new splits
    new_train.to_csv('train_split.csv', index=False)
    validation.to_csv('validation_split.csv', index=False)
    
    print(f"\n✓ Created train_split.csv: {len(new_train)} images")
    print(f"✓ Created validation_split.csv: {len(validation)} images")
    
    print(f"\n📊 Class distribution in new training set:")
    print(new_train['target'].value_counts().sort_index())
    
    print(f"\n📊 Class distribution in validation set:")
    print(validation['target'].value_counts().sort_index())
    
    print(f"\n⚠️  IMPORTANT: Update your training script to use:")
    print(f"   TRAIN_CSV = 'train_split.csv'")
    print(f"   TEST_CSV = 'validation_split.csv'")
    
    return new_train, validation


def check_test_images():
    """
    Check what's in your test folder and create a basic CSV
    """
    
    print("="*60)
    print("OPTION 3: Check test folder contents")
    print("="*60)
    
    test_dir = 'test'  # Update with your test directory
    
    if not os.path.exists(test_dir):
        print(f"❌ Directory '{test_dir}' not found!")
        print(f"Please update the 'test_dir' variable with the correct path.")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    test_images = []
    
    for file in os.listdir(test_dir):
        if any(file.endswith(ext) for ext in image_extensions):
            test_images.append(file)
    
    test_images.sort()
    
    print(f"\n✓ Found {len(test_images)} images in '{test_dir}'")
    print(f"\nFirst 10 images:")
    for img in test_images[:10]:
        print(f"  - {img}")
    
    # Check current test.csv
    if os.path.exists('test.csv'):
        test_df = pd.read_csv('test.csv')
        print(f"\n📄 Current test.csv:")
        print(f"   Rows: {len(test_df)}")
        print(f"   Columns: {list(test_df.columns)}")
        
        if 'target' not in test_df.columns:
            print(f"\n⚠️  test.csv does NOT have labels (target column)")
            print(f"   This CSV is for prediction only, not for training/evaluation.")
    
    return test_images


def main():
    """Main function with user menu"""
    
    print("\n" + "="*60)
    print("TEST CSV PREPARATION HELPER")
    print("="*60)
    
    print("\nWhat would you like to do?\n")
    print("1. I HAVE labels for test images → Create test.csv with labels")
    print("2. I DON'T have test labels → Split train.csv into train + validation")
    print("3. Just check what's in my test folder")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\n⚠️  You need to manually edit the script first!")
        print("Open this file and add your test image labels in the")
        print("create_test_csv_with_labels() function.\n")
        
        proceed = input("Have you added the labels? (y/n): ").strip().lower()
        if proceed == 'y':
            create_test_csv_with_labels()
        else:
            print("Please edit the script first, then run it again.")
    
    elif choice == '2':
        print("\nThis will create NEW files: train_split.csv and validation_split.csv")
        proceed = input("Continue? (y/n): ").strip().lower()
        if proceed == 'y':
            split_train_into_train_val()
    
    elif choice == '3':
        check_test_images()
    
    elif choice == '4':
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice. Please run again and select 1-4.")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("""
For a proper machine learning project, you need LABELED test data.

✅ BEST OPTION: Use Option 2 (split your labeled training data)
   - This gives you both train and test sets with labels
   - You can properly evaluate your model
   - This is standard practice when you don't have a separate test set

❌ Using test.csv WITHOUT labels:
   - You can't measure accuracy
   - You can't see confusion matrix
   - You can only make predictions (no evaluation)
    """)


if __name__ == '__main__':
    main()