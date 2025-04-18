# Advanced HPC-CI Webinar Series: Porting your AI application to Voyager Gaudi Architecture
## Tuesday, April 8, 2025
## 11:00 AM - 12:30 PM PDT
## Remote event

## Summary
This tutorial will cover how to run a Deep Learning (DL) model, based on the Pytorch framework, on the Voyager supercomputer. Voyager comprises an innovative architecture to optimize DL applications thanks to its Intel Habana Gaudi accelerators. We will discuss Voyager architecture and show how to launch jobs using containers in Kubernetes. Then, we will take a Pytorch application and will port it using Intel Gaudi libraries. We will discuss how to run Jupyter notebooks on Voyager. Finally, we will show how to pull a Hugginface model and run inference on Voyager using Transformers/Diffusers and Optimum-Habana libraries.

## Instructor
### Dr. Javier Hernandez-Nicolau
### Data Scientist, SDSC
Dr. Javier Hernandez-Nicolau is a Data Scientist in the Scientific applications group at SDSC. He obtained his PhD in Plasma Physics and Nuclear Fusion at the University of Carlos III of Madrid (Spain) in 2019 and then worked as a Project Scientist at the University of California-Irvine. He developed and optimized MHD and gyrokinetic fusion plasma codes for several HPC systems using Fortan, OpenMP and OpenACC. He joined SDSC in 2023 where he keeps collaborating with the Nuclear Fusion community developing ML/AI applications. He also offers user support to deploy a large variety of AI models on the Voyager machine which uses Intel Gaudi accelerators.

## Webinar material
In this repository you can find the files Javier showed during the webinar. Those are:
- An updated pdf with the slides for the webinar.
- Yaml files to launch the pods with the examples: hello world, mnist model and Jupyter Notebook.
- The mnist code with the CNN model ported to Intel Gaudi architecture on Voyager.
- A jupyter notebook with a Stable Diffusion model from Hugginface running with [optimum-habana](https://github.com/huggingface/optimum-habana) framework.

## Useful links
- [Intel Gaudi documentation](https://docs.habana.ai/en/latest/)
- [Tutorials and Reference models on Voyager](https://github.com/javierhndev/Voyager-Reference-Models)
- [Optimum-Habana](https://github.com/huggingface/optimum-habana) (to run Huggingface models on Voyager)
