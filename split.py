import sys

def split_file(input_file, train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
    # Validate that the ratios add up to 1
    if (train_ratio + dev_ratio + test_ratio) != 1.0:
        print("Error: Ratios must sum to 1.")
        return

    # Read the content of the original file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Calculate split indices
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    dev_end = train_end + int(total_lines * dev_ratio)

    # Split the lines according to calculated indices
    train_lines = lines[:train_end]
    dev_lines = lines[train_end:dev_end]
    test_lines = lines[dev_end:]

    # Output files
    output_base = input_file.rsplit('.', 1)[0]  # Remove file extension from input file
    train_file = f"{output_base}_train.txt"
    dev_file = f"{output_base}_dev.txt"
    test_file = f"{output_base}_test.txt"

    # Write to files
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(dev_file, 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print("Files created:")
    print(f"Train file: {train_file} ({len(train_lines)} lines)")
    print(f"Dev file: {dev_file} ({len(dev_lines)} lines)")
    print(f"Test file: {test_file} ({len(test_lines)} lines)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
    else:
        input_file = sys.argv[1]
        split_file(input_file)
