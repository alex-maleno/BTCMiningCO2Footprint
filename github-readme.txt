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

3. With everything committed, just type "git push" and you are all done! 
    - The first time you push is another instance where it may prompt you for 
    a username and password, see step 3 in the GitHub Setup for instructions

Getting the new code from the repository onto your computer:
All you have to do is type "git pull" within BTCMiningCO2Footprint (e.g. not in 
any sub-folders) and you should be good to go!

When you are working on a file that multiple people are working on, please
send a message in #github-push-pull on slack so that we can avoid "forks"/"branches" - 
a fork/branch happens when there are two versions of the same file with different 
modifications. I can always fix them, keeping the changes from both branches, but
its a little annoying and can lead to a slowdown