import os
def rename(folder_name):
    target = '%s%s' % ('./',folder_name)
    for filename in os.listdir(target):
        new_filename = filename.replace("B1", folder_name)
        filename = '%s%s%s' % (target,'/',filename)
        new_filename = '%s%s%s' % (target,'/',new_filename)
        os.rename(filename, new_filename) 



rename('B3')
