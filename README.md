[![version](https://img.shields.io/github/release-pre/a-mma/AquilaDB.svg)](https://github.com/a-mma/AquilaDB/releases) 
[![GitHub license Apache 2.0](https://img.shields.io/github/license/a-mma/AquilaDB)](https://github.com/a-mma/AquilaDB/blob/master/LICENSE)

[![docker build](https://img.shields.io/docker/cloud/automated/ammaorg/aquiladb.svg) ![docker build ststus](https://img.shields.io/docker/cloud/build/ammaorg/aquiladb.svg) ![docker stars](https://img.shields.io/docker/stars/ammaorg/aquiladb.svg) ![docker pulls](https://img.shields.io/docker/pulls/ammaorg/aquiladb.svg)](https://hub.docker.com/r/ammaorg/aquiladb)

[![github open issues](https://img.shields.io/github/issues-raw/a-mma/AquilaDB.svg) ![github closed issues](https://img.shields.io/github/issues-closed-raw/a-mma/AquilaDB.svg)](https://github.com/a-mma/AquilaDB/issues) [![github stars](https://img.shields.io/github/stars/a-mma/AquilaDB.svg?style=social)](https://github.com/a-mma/AquilaDB)

[![Build Status](https://img.shields.io/travis/a-mma/AquilaDB/develop?label=CI%20Tests&logo=travis)](https://travis-ci.org/a-mma/AquilaDB)

Do you like this project? We love getting a **star** ⭐ and **shout-out** 🗣️from you in return! 🤗

[Community support: ![discord chatroom for discussions](https://www.freeiconspng.com/minicovers/flat-discord-material-like-icon--2.png)](https://discord.gg/5YP7zHS)

**[Documentation](https://aquiladb.xyz/docs/introduction)**

# AquilaDB
**AquilaDB** is a vector database to store **Feature Vectors** along with **JSON Document Metadata**. Do **k-NN** retrieval from anywhere, even from the darkest rifts of Aquila (in progress). It is dead simple to set up, language agnostic and drop in addition for your Machine Learning Applications. AquilaDB, as of current features is ready solution for Machine Learning engineers and Data scientists to build **[Neural Information Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/INR-061-Mitra-neuralir-intro.pdf)** applications out of the box with minimal dependencies (visit [wiki page](https://github.com/a-mma/AquilaDB/wiki) for use case examples).

AquilaDB 1.0 release is a distant goal to achieve. Visit **contribute** section below to see detailed development plan and milestones. 
We make sure that each `release` and `AquilaDB Master branch` are stable with all features planned up to date. All new pull requests are made to `develop` branch. So, `develop` is the default and bleeding edge branch with all the latest updates.

[Github](https://github.com/a-mma/AquilaDB), [Docker Hub](https://hub.docker.com/r/ammaorg/aquiladb), [Documentation (dedicated Wiki page)](https://github.com/a-mma/AquilaDB/wiki)


# Who is this for

* If you are working on a data science project and need to store a hell lot of data and retrieve similar data based on some feature vector, this will be a useful tool to you, with extra benefits a real world web application needs.
* Are you dealing with a lot of images and related metadata? Want to find the similar ones? You are at the right place.
* If you are looking for a document database, this is not the right place for you.

# Technology
AquilaDB is not built from scratch. Thanks to OSS community, it is based on a couple of cool open source projects out there. We took a couch and added some wheels and jetpacks to make it a super cool butt rest for Data Science Engineers. While **CouchDB** provides us network and scalability benefits, **FAISS** and **Annoy** provides superfast similarity search. Along with our peer management service, AquilaDB provides a unique solution.

# Prerequisites
You need `docker` installed.

# Usage

AquilaDB is quick to setup and run as docker a container. All you need to do is either build it from source or pull it from Docker hub.

#### Option 1: build from source
* clone this repository
* build image: `docker build -t ammaorg/aquiladb:latest .`
#### Option 2: pull from dockerhub
* pull image: `docker pull ammaorg/aquiladb:latest`
#### Finally, deploy
* deploy: `docker run -d -i -p 50051:50051 -v "<local data persist directory>:/data" -t ammaorg/aquiladb:latest`

# Client SDKs
We currently have multiple client libraries in progress to abstract the communication between deployed AquilaDB and your applications.

[Python](https://github.com/a-mma/AquilaDB-Python)

[Node JS](https://github.com/a-mma/AquilaDB-NodeJS)

AquilaDB exposes [gRPC](https://grpc.io/) APIs for the clients. Which means, you can communicate directly to AquilaDB from your favourite language ([API reference](https://github.com/a-mma/AquilaDB/tree/develop/src/proto)). Above clients makes use of that to abstract the communication details from end user. If you are familiar with gRPC and would like to contribute a new client library in any other language, please let us know.
Protocol buffers [API reference](https://github.com/a-mma/AquilaDB/blob/master/src/proto/vecdb.proto).
[Example usage](https://github.com/a-mma/AquilaDB/blob/master/src/test/client.js) of APIs in node js. 

# Benchmark
For benchmark results, visit https://aquiladb.xyz/docs/adb-benchmarks

# Progress
This project is still under active development (pre-release). It can be used as a standalone database now. Peer manager is a work in progress, so networking capabilities are not available now. With release v1.0 we will release pre-optimized version of AquilaDB.

# Contribute
We have [prepared a document](https://docs.google.com/document/d/1bT2_9FQIxQpx_rdYbkTukn_DJRi_haVK_ixTf8uTaDE/edit?usp=sharing) to get anyone interested to contribute, immediately started with AquilaDB.

Here is our high level [release roadmap](https://user-images.githubusercontent.com/19545678/62313851-5af82880-b4af-11e9-84f6-21e24bf46e8a.png).

# Learn

We have started meeting developers and do small talks on AquilaDB. Here are the slides that we use on those occasions: http://bit.ly/AquilaDB-slides 

**Video:**

[<img alt="introduction to Neural Information retrieval with AquilaDB" src="http://img.youtube.com/vi/-VYpjpLXU5Q/0.jpg" width="300" />](http://www.youtube.com/watch?v=-VYpjpLXU5Q)

As of current AquilaDB release features, you can build **[Neural Information Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/INR-061-Mitra-neuralir-intro.pdf)** applications out of the box without any external dependencies. Here are some useful links to learn more about it and start building:

* These use case examples will give you an understanding of what is possible and what not: https://github.com/a-mma/AquilaDB/wiki
* Microsoft published a paper and youtube video on this to onboard anyone interested: 
  * paper: https://www.microsoft.com/en-us/research/uploads/prod/2017/06/INR-061-Mitra-neuralir-intro.pdf
  * video: https://www.youtube.com/watch?v=g1Pgo5yTIKg
* Embeddings for Everything: Search in the Neural Network Era: https://www.youtube.com/watch?v=JGHVJXP9NHw
* Autoencoders are one such deep learning algorithms that will help you to build semantic vectors - foundation for Neural Information retrieval. Here are some links to Autoencoders based IR:
  * go to chapter 15 in this link: https://www.cs.toronto.edu/~hinton/coursera_lectures.html
  * https://www.coursera.org/lecture/ml-foundations/examples-of-document-retrieval-in-action-CW25H
  * https://www.coursera.org/lecture/intro-to-deep-learning/autoencoders-101-QqBOa
* Note that, the idea of information retrieval applies not only to text data but for any data. All you need to do is, encode any source datatype to a dense vector with deep neural networks.

# License

Apache License 2.0 [license file](https://github.com/a-mma/AquilaDB/blob/master/LICENSE)

created with ❤️ a-mma.indic (a_മ്മ)
