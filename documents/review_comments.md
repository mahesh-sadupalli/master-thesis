# Review Comments — Chapter 1 (report_ch1.pdf)

**Reviewer:** Abhishek Dhiman (Mentor)

**Total comments:** 30

---

## Page 2

### Comment 1

**Comment:** Rephrase: We are using the Hybrid InSitu InTransit approach from Kitware. There are three different approaches which are updated versions: InSitu -> InTransit -> Hybrid InSitu InTransit

---

### Comment 2

**Highlighted text:** *"face significant I/O bottlenecks and scalability limitations [3, 4]. This work proposes
an in-situ and in-transit compression framework [5, 6] that employs neural networks
to learn compact representations of data during runtime."*

**Comment:** Rephrase: We are using the Hybrid InSitu InTransit approach from Kitware. There are three different approaches which are updated versions: InSitu -> InTransit -> Hybrid InSitu InTransit

---

### Comment 3

**Highlighted text:** *"explores two complementary neural network architectures"*

**Comment:** Rephrase: Do not quantify now, mention multiple approach for online. But offline and online is explained briefly to highlight bottle neck

---

### Comment 4

**Highlighted text:** *"fields"*

**Comment:** Data

---

### Comment 5

**Highlighted text:** *"temporal windows"*

**Comment:** Not to mention in abstract, but in methodology. Its like a configurable feature

---

### Comment 6

**Highlighted text:** *"Initial experiments with coordinate-based MLPs across three model sizes (base:
6,692 parameters, medium: 14,644, large: 25,668) demonstrate compression ratios
from 7,713:1 to 27,395:1 with offline Peak Signal-to-Noise Ratio (PSNR) of 32–36 dB
and Structural Similarity Index Measure (SSIM) of 0.955–0..."*

**Comment:** Repharase: Do not put exact values, but use increment percentage like 6692 params be X, medium is 2.5x, large is 5x. For PSNR and SSIM since there is not standard value to be achieved so just mention these as evaluation metrics [MSE, MAE, PSNR, SSIM]

---

### Comment 7

**Highlighted text:** *"temporal domain"*

**Comment:** configurable parameter

---

### Comment 8

**Highlighted text:** *"validated o"*

**Comment:** evaluated for vortex shedding case which is a benchmarking CFD simulation 
1. https://innovationspace.ansys.com/courses/wp-content/uploads/sites/5/2020/08/Unsteady-flow-over-a-cylinder-Results-and-Discussion.pdf

2. https://link.springer.com/article/10.1007/s10494-008-9186-7

---

### Comment 9

**Highlighted text:** *"streaming datasets in real-time"*

**Comment:** large scale spatio-temporal data streamed from a running simulation.

---

### Comment 10

**Highlighted text:** *"Abstract"*

**Comment:** the input data is a point cloud, not images or structured data. this can also be mentioned in the abstract.

---

### Comment 11

**Comment:** Also, generalize with other domain (in abstract, onwards to CFD) not just CFD. Basically, any pointcloud, numeric, and streaming data

---

## Page 7

### Comment 12

**Highlighted text:** *"Introduction"*

**Comment:** Looks good

---

### Comment 13

**Highlighted text:** *"modern High Performance Computing (HPC) environments [11]. First, writing full-
resolution data to disk can dominate a simulation’s total wall-clock time, a problem
known as the Input/Output (I/O) bandwidth gap [4, 6].
Second, finite storage
capacity forces scientists to discard temporal or spatial ..."*

**Comment:** Here, need for scientific and advance visualization capability can also be added. 
This would complete the process from data generation, storage, regular feedback, result visualization(individual, collaborative, business PoV, etc)

---

## Page 8

### Comment 14

**Highlighted text:** *"8]. Current compression solutions each fall short in this setting: lossless methods pro-
vide insufficient reduction [23]; error-bounded lossy compressors such as SZ [24, 25]"*

**Comment:** Also, u compress and decompress all the data, else more files that increases I/O

---

### Comment 15

**Highlighted text:** *"currently with running simulations, achieving significant data
reduction while maintaining reconstruction accuracy sufficient"*

**Comment:** significant data reduction is difficult with numeric data. we can remvoe this part

---

## Page 9

### Comment 16

**Highlighted text:** *"This section reviews the principal areas of prior work relevant to this thesis, identi-
fying the research gap that motivates the proposed approach."*

**Comment:** it gernerally good to have more text between parent and child heading in order to brief ech sub heading

---

### Comment 17

**Highlighted text:** *"The I/O bottleneck has driven frameworks for concurrent data processing. In-situ
approaches such as ParaView Catalyst [5, 42], VisIt LibSim [43], SENSEI [12], and
Ascent [13, 44] embed analysis within the simulation’s memory space. In-transit sys-
tems such as ADIOS 2 [6] and Damaris [4] offload pro..."*

**Comment:** nice

---

## Page 10

### Comment 18

**Highlighted text:** *"veals that no existing work combi"*

**Comment:** rephrase it: looks very deterministic: maybe update to 'like existing approaches does check all the boxes'

---

## Page 11

### Comment 19

**Highlighted text:** *"The critical gap lies at the intersection of these areas: no existing frame-
work combines coordinate-based neural compression with incremental"*

**Comment:** rephrase

---

### Comment 20

**Highlighted text:** *"alidate a coordinate-based"*

**Comment:** is there any definition for coordinate based NN (check other mentions also)?. else just say spatio-temporal data

---

### Comment 21

**Highlighted text:** *"Establish a comprehensive evaluation framework: Define and apply
multi-metric evaluation including compression ratio, Mean Squared Error (MSE"*

**Comment:** can also add MAE mean absolute error as a quality metric (SSIM, PSNR, MAE) and MSE for loss function or which ever u find good

---

### Comment 22

**Highlighted text:** *"Validate"*

**Comment:** rephrase, validate is key term in CFD and used in more specific context

---

## Page 12

### Comment 23

**Highlighted text:** *"RQ1:
How can coordinate-based neural network architectures and
training protocols be designed to effectively learn compact representa-
tions of streaming spatio-temporal data, and what is the quantitative"*

**Comment:** keep the questions short 2 lines kind off as u are also explaining the expectation from the answer

---

### Comment 24

**Highlighted text:** *"framework with three model sizes"*

**Comment:** 3 different capacity are like verification and u compare quality metric from these 3 and check how much your metric is changing when u increase th eNN capacity then u pick one to try it on a different dataset

---

## Page 13

### Comment 25

**Highlighted text:** *"sion of streaming data. The approach is evaluated both on recent temporal
windows and on the full dataset, characterising the catastrophic forgetting"*

**Comment:** rephrase : keep the name consistent for sliding temporal window and define it also what do u mean by that, meaning if its configurable parameters or not, how many timesteps u have take and things like tha

---

### Comment 26

**Highlighted text:** *"Comprehensive dual-metric evaluation framework: Establishment of a
multi-metric evaluation protocol combining compression ratio, MSE, PSNR,"*

**Comment:** emphasis on training and test time also, because the approach needs to be integration with a running simulation so it will give a better view

---

### Comment 27

**Highlighted text:** *"elling, structural mechanics, electromagnetics, molecular dynamics), experimental
validation is performed on a single CFD dataset. Generalisation to other domains"*

**Comment:** rephrase

---

### Comment 28

**Highlighted text:** *"not experimentally"*

**Comment:** rephrase

---

### Comment 29

**Highlighted text:** *"Hardware and deployment scope: Experiments are conducted using GPU-
accelerated training on a single compute node."*

**Comment:** little mor info would be better

---

### Comment 30

**Highlighted text:** *"Scalability: The framework is validated on datasets with approximately 107
spatio-temporal samples.
Scalability to problems with 109 or more points per
timestep, or to three-dimensional spatial domains, requires further investigation."*

**Comment:** 10⁷ is all the datapoints in the csv file ?, how much is fed to the NN for incremental learning (temporal window with/without batching ) and offline (batch)

---

