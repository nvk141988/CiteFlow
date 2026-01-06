import ast
import json
import os

def main():
    file_path = os.path.join("Results", "cr_analysis_list2.txt")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    count_ref = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            #For each line
            for line in f:
                content = line
                
                # Parse the string content as a Python literal (dictionary)
                data = ast.literal_eval(content)
                
                # Print the data pretty-printed
                json_output = json.dumps(data, indent=4)
                print(json_output)
                
                output_path = os.path.join("Results", f"ref_{count_ref}.json")
                count_ref += 1
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(json_output)
                print(f"\nOutput saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
