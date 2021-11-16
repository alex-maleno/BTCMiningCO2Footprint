Setting up GitHub on the command line:

1. Set up your GitHub Token - similar to a password, but meant to expire
after a shorter time. I set mine for 60 days, essentially just to use 
for this project. Follow instructions here:
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
    - Make sure you save this token in a safe place! It acts as your password
    to interface with the GitHub API on the command line, and you will need
    it later

2. Go on your command line and navigate to the place where you want the
repository to be saved - I put it into my folder for this class
    - You can use the command "cd <folder name>" to navigate to a folder
    in the command line
    - You don't need to make a folder specifically for the repository: when
    you clone the repository it will be saved in a folder called 
    BTCMiningCO2Footprint, the repository name

3. Copy and paste the following command into the command line:
git clone https://github.com/alex-maleno/BTCMiningCO2Footprint.git
    - You may be prompted to enter your username and password at this step, or
    it may prompt you later. Once prompted, enter your GitHub username and the 
    token you generated in step 1 

4. You are all set! If you type "ls" in the command line, you should see the folder
named "BTCMiningCO2Footprint"

GitHub Commands:

Now that you have cloned the repository to your local machine, you have a copy
of all the code that is in the repository. This does not update automatically.

Adding code to the repository:

Whenever you make any changes to existing code or add new files, you need to 
"push" them to the repository.
1. Within the folder you are working in, type "git add --all"
    - this adds all the files that are modified within this folder, and if you
    are working in a new folder, it will add that new folder to the repository.
    Unfortunately, if you have multiple folders within a folder, you may need
    to do these steps multiple times 

2. Now that all the files are added to the queue to push, type 
"git commit -m 'message to commit' ", where 'message to commit' briefly describes
the changes that you are adding to the repository, and should be contained within
double quotes (""), I just used single quotes to make it more clear in the command
    - After committing, you should see a message talking about what files have 
    been modified/created/deleted. You only need to worry about this if there is
    some indication that it failed, and if it does lmk

3. With everything committed, just type "git push" and you are all done! 
    - The first time you push is another instance where it may prompt you for 
    a username and password, see step 3 in the GitHub Setup for instructions

After pushing, you should see some message indicating how much has been changed and 
if the push failed. Lmk if it fails, or google the error message.

Getting the new code from the repository onto your computer:

All you have to do is type "git pull" within BTCMiningCO2Footprint (e.g. not in 
any sub-folders) and you should be good to go!

GitHub Usage Tips and Tricks:

- When you are working on a file that multiple people are working on, please
send a message in #github-push-pull on slack so that we can avoid "forks"/"branches" - 
a fork/branch happens when there are two versions of the same file with different 
modifications. I can always fix them, keeping the changes from both branches, but
its a little annoying and can lead to a slowdown. A good rule of thumb to follow is that
you should always pull before you start working.
- If you are using a text editor, many times there are extensions to add GitHub to your
editor. For example, in VS Code, the "GitHub Pull Requests and Issues" extension will
highlight the line numbers that are edited from the file version in the repository as
well as any new lines that you added, so that you can quickly see your progress and 
remember to push your changes at the end of your coding session.

Useful Command Line Commands:

pwd - "point working directory", prints out the path to where you are currently
      working
cd <path> - changes your directory (or folder) to the specified folder or path.
            This works with paths relative to your current working directory,
            or with absolute paths (starting from your root directory)
          - to go "up" or back a directory, type "cd .."
ls - lists the contents of the directory that you are currently in
touch <file name> - creates a new empty file with <file name> in your current
                    working directory that you can then open in the text editor
                    of your choice
mkdir <directory name> - creates a new directory (or folder) with <directory name>
rm <file name> - deletes the <file name> specified
rm -r <folder name> - deletes the <folder name> specified along with everything
                      in it, recursively
cat <file name> - quickly prints out the contents of <file name> in the command
                  line, so that you don't need to open it in a text editor. This
                  is for viewing only, useful to see what an input file looks like
