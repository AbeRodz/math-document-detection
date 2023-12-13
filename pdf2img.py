import fitz
import os
import pandas as pd

DIR  = "./dataset/Material por ID/"
FILTERED_DIR = "./dataset/filtered/" 
OUTPUT_DIR = "dataset/images_pool/"


def get_labels(df : pd.DataFrame, file_name : str) : 
        
        return tuple(df[df['Nombre'] == file_name].to_numpy().flatten()[1:-2])

def get_file_name(file_path: str) -> str:
    dir_name = file_path.split('/')[-1]
    dir_name = dir_name.split('.pdf')[0]
    return dir_name

def pdf2_png(file_path:str, directory :str, file_name : str):
    
    pdf_document = fitz.open(file_path)

    for page in pdf_document:
        pix = page.get_pixmap()
    
        pix.save(f"{directory}/{file_name}-page-%i.png" % page.number)

    pdf_document.close()




def process_pdf(file_name, df):
    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(FILTERED_DIR, file_name)
        dir_name = get_file_name(pdf_path)

        # Convert PDF to images and extract labels
        pdf2_png(pdf_path, OUTPUT_DIR, dir_name)
        labels = get_labels(df, f"{dir_name}.pdf")

        return {
            'ImageName': dir_name,
            'Level': labels[0],
            'University': labels[1],
            'Math Subject': labels[2],
            'Course': labels[3],
            'Type': labels[4],
            'Annotation': labels[5]
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Script that transforms a pdf file onto a set images per page.")

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="input file path",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="output file path",
        required=True
    )

    args = parser.parse_args()

    pdf2_png(file_path=args.input_path, directory= args.output_path)