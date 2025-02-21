import os

def convert_usfm_to_txt(folder_path):
    """
    Convert USFM files numbered 41 to 67 in the specified folder to TXT format.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Create an output folder for the TXT files
    output_folder = os.path.join(folder_path, 'converted_txt')
    os.makedirs(output_folder, exist_ok=True)

    # Loop through files numbered 41 to 67
    for number in range(41, 68):
        input_filename = f"{number}.usfm"
        input_filepath = os.path.join(folder_path, input_filename)

        if os.path.exists(input_filepath):
            with open(input_filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()

            # Remove USFM markers (e.g., \id, \v, \c)
            content_clean = remove_usfm_markers(content)

            # Save as TXT
            output_filename = f"{number}.txt"
            output_filepath = os.path.join(output_folder, output_filename)

            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                outfile.write(content_clean)

            print(f"Converted: {input_filename} -> {output_filename}")
        else:
            print(f"File not found: {input_filename}")

def remove_usfm_markers(content):
    """
    Basic function to remove USFM markers starting with backslashes (e.g., \v, \c).
    """
    import re
    # Remove lines starting with a backslash
    content = re.sub(r'\\[a-zA-Z0-9]+\s?', '', content)
    return content.strip()

if __name__ == "__main__":
    # Specify the folder where your files are located
    folder_path = os.path.expanduser('~/Desktop/test/experiment')
    convert_usfm_to_txt(folder_path)