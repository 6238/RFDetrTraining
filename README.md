This repo contains the training code for the RF-Detr models we use on the robot.

It uses Vertex AI and a Cloud Storage Bucket so you will need to setup a google cloud console project and paste the service worker json file into this repo.

Upload Train App Tarball:
```
python setup.py sdist --formats=gztar
```

In google cloud sdk shell:
```
gsutil cp dist/trainer-0.1.tar.gz gs://YOUR_BUCKET/packages/trainer-0.1.tar.gz
```