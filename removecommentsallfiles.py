import sys, re
import os
for path, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            file_no_comments=file + '_no_comments.py'
            file_no_comments=os.path.join(path,file_no_comments)
            file_object=open(os.path.join(path,file),'r')
            file_object_no_comment=open(file_no_comments,'w')
            codeline=file_object.read()
            codeline_no_comment=re.sub('""".*?"""','\n',codeline,flags=re.DOTALL|re.MULTILINE)
            codeline_no_comment=re.sub('\n\n\n','\n',codeline_no_comment,flags=re.DOTALL|re.MULTILINE)
            codeline_no_comment=re.sub('\n\n','\n',codeline_no_comment,flags=re.DOTALL|re.MULTILINE)
            file_object_no_comment.write(codeline_no_comment)
            file_object.close()
            file_object_no_comment.close()
            print(os.path.join(path, file))
