import os

def save_uploaded_files(file):
    # Save the file to a desired location
    # current_directory = os.getcwd()
    # parent_directory = os.path.dirname(current_directory)   
    # pdf_folder = os.path.join(parent_directory, 'database/uploads') # Update with the folder containing your PDF files
    # file.save(pdf_folder)
    file.save('/Users/zade/Downloads/github.com/mehrdad-zade/portfolioGPT/database/uploads/'+file.filename)
    return 'saved'
