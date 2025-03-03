# Accelerating Software Vulnerability Assessment Model Construction through Code Simplification

## Introductin
CS-SVA (Code Simplification for Software Vulnerability Assessment) is a novel framework designed to improve the efficiency of vulnerability detection models by simplifying input code while preserving key vulnerability-related features. This approach employs a dual-granularity pruning strategy, integrating statement-level and token-level simplifications, significantly reducing computational costs without substantial performance degradation.

By leveraging pre-trained language models (PLMs), such as CodeT5, CS-SVA achieves a balance between model accuracy and inference efficiency, making it suitable for large-scale vulnerability assessment tasks.


## Approach
![Framework](figs/framework.png)


## Dataset
We utilize the **MegaVul** dataset for vulnerability assessment. You can access the original dataset from its official repository on GitHub: [MegaVul](https://github.com/Icyrockton/MegaVul).  

For convenience, the processed version of the dataset used in our experiments is available for download from **Google Drive**: [Download Here](https://docs.google.com/spreadsheets/d/1Ovd8CkY89f2u-6P-2wGKF-xMYptS3kBO/edit?usp=sharing&ouid=111461340104776755635&rtpof=true&sd=true).


## Requriments
To set up the required dependencies for this project, install the necessary packages by running the following command:  

```bash
pip install -r requirements.txt


## 

