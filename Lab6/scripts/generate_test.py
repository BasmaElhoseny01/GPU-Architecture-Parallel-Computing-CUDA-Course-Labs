import random
import sys

def generate_sorted_floats_file(filename, num_floats):
    floats = [random.uniform(-10.0, 10.0) for _ in range(num_floats)]
    floats.sort()
    with open(filename, 'w') as file:
        for float_val in floats:
            file.write(f"{float_val:.2f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_sorted_floats.py <folder_path> <num_floats>")
        sys.exit(1)

    folder_path = sys.argv[1]
    num_floats = int(sys.argv[2])

    # Generate and save the sorted floats to file
    file_path=folder_path+'/'+str(num_floats)+".txt"
    generate_sorted_floats_file(file_path, num_floats)
    print(f"Generated {num_floats} sorted floats in '{file_path}'.")


# python ./scripts/generate_test.py ./tests 10000
