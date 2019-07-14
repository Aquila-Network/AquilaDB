FROM ubuntu:18.04
MAINTAINER a_mma

ADD ./ ./AquilaDB

# install couchdb
RUN apt-get update; apt-get install -y curl; apt-get install -y --no-install-recommends git; \
	echo "deb https://apache.bintray.com/couchdb-deb bionic main" \
    | tee -a /etc/apt/sources.list; \
    apt-get install -y --no-install-recommends gnupg; \
    curl -L https://couchdb.apache.org/repo/bintray-pubkey.asc \
    | apt-key add -; \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends couchdb; \
	apt-get purge -y --auto-remove gnupg; \
#
#	clone repository
#	git clone https://github.com/a-mma/AquilaDB.git; \
#
#   setup node environment
	curl -sL https://deb.nodesource.com/setup_10.x | bash -; \
	apt install -y --no-install-recommends nodejs; \
	apt-get install -y --no-install-recommends make; \
	cd /AquilaDB/src && rm package-lock.json || true && npm install; \
	npm install pm2 -g; \
	npm cache clean --force; \
	apt-get purge -y --auto-remove curl make git; \
#
# setup python environment
	python3 --version; \
	apt-get -y install python3-pip wget --no-install-recommends; \
	wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc; \
	export PATH=/opt/conda/bin:$PATH; \
	conda config --set auto_update_conda False ; \
	conda create -n myenv python && conda install faiss-cpu=1.5.1 -c pytorch -y; \
	python -m pip install grpcio-tools; \
	apt-get purge -y --auto-remove wget; \
	conda clean --all; \
	rm -r "/opt/conda/pkgs/"; \
	rm -rf /var/lib/apt/lists/* \
#
# make init script executable
	chmod +x /AquilaDB/src/init_aquila_db.sh

CMD /AquilaDB/src/init_aquila_db.sh && tail -f /dev/null
