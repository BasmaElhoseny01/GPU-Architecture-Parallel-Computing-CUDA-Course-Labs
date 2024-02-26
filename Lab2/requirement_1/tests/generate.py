
import argparse
import random

def generate_test_case(rows, columns):
    test_case = f"{rows} {columns}\n"
    for _ in range(rows * 2):
        row = " ".join(str(round(random.uniform(1, 10), 1)) for _ in range(columns))
        test_case += f"{row}\n"
    return test_case

def generate_test_file(num_test_cases, rows, columns,filename):
    with open(filename, "w") as file:
        file.write(f"{num_test_cases}\n")
        for _ in range(num_test_cases):
            test_case = generate_test_case(rows, columns)
            file.write(test_case)

def main():
    parser = argparse.ArgumentParser(description="Generate a test file with random matrices.")
    parser.add_argument("num_test_cases", type=int, help="Number of test cases")
    parser.add_argument("rows", type=int, help="number of rows")
    parser.add_argument("columns", type=int, help="number of columns")
    # parser.add_argument("min_columns", type=int, help="Minimum number of columns")
    # parser.add_argument("max_columns", type=int, help="Maximum number of columns")
    parser.add_argument("filename", type=str, help="Output filename")
    args = parser.parse_args()

    generate_test_file(args.num_test_cases, args.rows, args.columns,args.filename)

if __name__ == "__main__":
    main()


# python ./tests/generate.py num_test_cases rows columns filename
#  python ./tests/generate.py 1 4 3 ./tests/test_4_3.txt