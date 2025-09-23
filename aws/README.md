# AWS Files README

## Hadoop Service

```sh
sudo cp hadoop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hadoop.service
sudo systemctl start hadoop.service
```

## Install AWS Cli V2 on existing

```bash
unzip awscliv2.zip
sudo ./aws/install --install-dir /opt/pytorch/aws-cli-v2 --bin-dir /opt/pytorch/bin
```
