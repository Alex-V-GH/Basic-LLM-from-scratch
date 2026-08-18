#Basic LLM From Scratch
or
#Road to Rosa

#Introduction:
This project comes from needing an llm agent to integrate in Violetta (check my other repos). I tried to use some of the most common, but none of them was able to cover the need.
So it all began: I tried to finetune some specific model, but failed due to not having too much information about it´s architecture. 
Then i thought of making this little piece which i would have absolute (or at least much more) control over it.
I defined the goals for this project, which are: 
*It HAS to be a language model. Moreso, a LARGE LANGUAGE MODEL.
*It HAS to understand some specific orders given by the Violetta pipeline, and act according to them.
*It HAS to be relatively light, so it runs in most computers while the users can still use them.
*I HAVE to know EXACTLY the architecture of it, and change it as Violetta requires so.
*It HAS to "know" spanish as it´s main language, and english as secondary.
*It HAS to have conversational capability
*It HAS to manage context dynamically
*It HAS to learn some things from the user, and sometimes finetune itself to integrate that
*It HAS to know wether is the prompt an order, a conversation, a question, etc, as Violetta requires so.

#Current Status
Currently trying to finish the main pipeline with flawed results, but pointing to the goals

#Architecture
