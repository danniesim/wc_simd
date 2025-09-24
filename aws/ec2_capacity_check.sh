#!/bin/bash

aws ec2 run-instances   --instance-type p5.48xlarge   --image-id ami-0916509d3ddc82e36   --count 1   --dry-run