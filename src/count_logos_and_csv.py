import os
import csv

def count_logos(directory="logos"):
    # List all files in the directory
    files = os.listdir(directory)
    image_files_png = [file for file in files if file.endswith(('.png'))]
    image_files_favicon = [file for file in files if file.endswith(('_favicon.png'))]

    # Return the count of image files
    return len(image_files_png),  len(image_files_favicon)

def save_logo_csv(dir = 'logos', output_csv = 'logo_mapping.csv'):
    with open(output_csv, mode='w', newline='') as file:
        #Csv file with domain and associated logo_file in 'logos' dir
        writer = csv.writer(file)
        writer.writerow(['domain','logo_file'])

        for filename in os.listdir(dir):
            if filename.endswith(('.png','_favicon.png')):
                domain = filename.split('.')[0]
                domain = domain.replace("_", ".")
                writer.writerow([domain,filename])


if __name__ == "__main__":
    logo_count_png, logo_count_favicon = count_logos()
    save_logo_csv()
    print(f"There are {logo_count_png} logos (png) and {logo_count_favicon} favicons in the 'logos' directory.")



