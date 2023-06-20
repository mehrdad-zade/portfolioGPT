import os
from mySecrets import LOCAL_PATH

def save(file):
    # Save the file to a desired location
    # current_directory = os.getcwd()
    # parent_directory = os.path.dirname(current_directory)   
    # pdf_folder = os.path.join(parent_directory, 'database/uploads') # Update with the folder containing your PDF files
    # file.save(pdf_folder)
    file.save(LOCAL_PATH + 'database/uploads/'+file.filename)
    return 'saved'
