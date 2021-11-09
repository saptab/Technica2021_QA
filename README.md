# Technica2021_QA

Have you wondered how your smartphone answers questions (and when it sometimes can't)? Computers understand text through representations, including the application of question answering: taking a piece of text and finding the right answer. 

You will be developing a question answering system based on representation learning of the text as a part of [Technica 2021](https://inclusion.cs.umd.edu/events/techresearch).

We will be using different forms of representation like tf-idf (term frequency-inverse document frequency), DAN (Deep averaging networks) and distributional semantics using word2vec.

Based on the time available, we will also generate BERT (bi-directional encoder representations from transformers) and DPR representations from the provided text.

You have to convert the texts in the question and answer to representations that is understandable by the computer, find the closest answer to the question representation, and then decide whether you should trust this answer or not. You can see a similar system built by undergraduates take on Jeopardy! champions [here](https://www.youtube.com/watch?app=desktop&v=vH8cUGFOwPk&list=PLegWUnz91WfscfMYZvuSWEweBbIpfFDlF&index=20&t=0s).

# Concepts

Please refer to the [materials](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/) here for any quick reference. The important topics are highlighted below.

## Motivation for Question Answering

1. Importance of Question Answering in Civilization [Video] (https://www.youtube.com/watch?v=ziX5MJ4VKY0) [Slides] (https://docs.google.com/presentation/d/1CbmMbK71925ZztUh5W6dH58377ojHOhKMlTVNbTvVgg/edit#slide=id.p)

2. Turing Test & Question Answering [Video](https://www.youtube.com/watch?v=bdQIUv9FutE)

3. IBM Watson on Jeopardy [Video](https://www.youtube.com/watch?v=WCIFUJ5oeRA) [Slides](https://docs.google.com/presentation/d/13Cb7Bxz2NUGOPlO9ydAlpX1Y-zTcCo6k-wBvISPiruM/edit#slide=id.p)

## Feature Engineering

1. Introduction [Video](https://www.youtube.com/watch?v=4-se75AuETA)

2. Relevance with Question Answering [Video](https://www.youtube.com/watch?v=IzKFgigocAg) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/DATA_DIGGING/lecture_07.pdf)

## Tf-idf (term frequency - inverse document frequency) Vector Representations

1. Introduction: [Video] (https://www.youtube.com/watch?v=U6KgqeJkhU0) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/01a_ir.pdf) 

2. Tf-idf and Vector Models: [Video](https://www.youtube.com/watch?v=A5ounv0D_cs) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/01b_tfidf.pdf) 

3. Evaluation: [Video](https://www.youtube.com/watch?v=BxAzuCSvF8s) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/01c_evaluation.pdf) 

4. Example Problem on tf-idf: [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/01d_tfidf_ex.pdf) 

5. Example of a vector: X axis (dimension_1) -> i Y axis (dimension_2) -> j
	- Point (2,3) can be represented as a vector 2i+3j = u

	- Point (-2,3) can be represented as a vector -2i+3j = v

	- Dot Product (i.i=1, j.j=1, i.j=j.i=0 (e.g. u.v = (2i+3j).(-2i+3j) = -4+9 = 5)

	- Dimensionality to be the vocabulary size (12 words in the vocabulary, so vector dimensions will be 12.

	- The sky is blue (0.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0,0,0,0,0) = s1

	- The sun is bright today (0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.2,0,0,0,0,0) = s2

	- s1.s2 = 0.05 (They are not similar)

## Deep Averaging Networks (DAN)

1. Pytorch [Videos](https://www.youtube.com/watch?v=AZwwDIV2vcI) [Slides](https://www.youtube.com/watch?v=AZwwDIV2vcI)

2. DAN [Videos](https://www.youtube.com/watch?v=losFCNJbnZY) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/05e_dan.pdf)

## Distributional semantics using word2vec

1. Introduction: [Video] (https://www.youtube.com/watch?v=vErGaMc80WM) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/06a_distsim.pdf) 

2. Word2vec: [Video](https://www.youtube.com/watch?v=QyrUentbkvw) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_470/06b_word2vec.pdf)

3. Evaluation: [Video](https://www.youtube.com/watch?v=Jid_EVBoVLU) [Slides](http://users.umiacs.umd.edu/~jbg/teaching/CMSC_723/word2vec.pdf) 

## BERT (Bi-directional Encoder Representations from Transformers)

1. BERTology [Video](https://www.youtube.com/watch?v=YgfE-9N7Rio)

2. Understanding BERT [Video](https://www.youtube.com/watch?v=9Z7mN7ebWDA) [Slides](https://docs.google.com/presentation/d/1Wmte5rM9qgN-JHq34LFX8y3G2HebHKWRV9iu-p_fC2k/edit#slide=id.p)

3. BERT in Question Answering [Paper](https://arxiv.org/abs/1901.08634) [Slides](https://docs.google.com/presentation/d/17g-FG5jqvhQVGSkLI1BH3d7T2prTGTHcd3CeG7EDeVA/edit#slide=id.p)

## DPR (Dense Passage Retrieval)

1. Introduction [Paper](https://arxiv.org/abs/2004.04906) [Tutorial](https://towardsdatascience.com/how-to-create-an-answer-from-a-question-with-dpr-d76e29cc5d60)

# Installation Requirements

Python, PyTorch, NLTK, Numpy, word2vec, pandas and Sklearn python libraries. Use a code IDE like VSCode or PyCharm in case you do not have a Linux terminal on your machine. Alternatively install [Windows Terminal](https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701#activetab=pivot:overviewtab) on your laptop.

# Datasets

## Input Data
    https://github.com/Pinafore/nlp-hw/tree/master/data
    https://sites.google.com/view/qanta/resources?authuser=0 (Filtered Wikipedia JSON)
    https://github.com/Pinafore/cl1-hw/tree/master/project
## Example Output
   [Mock file](https://drive.google.com/file/d/1Z08Edd3UnrTeEVE0oP7n8zL5bkwLiW2a/view?usp=sharing)