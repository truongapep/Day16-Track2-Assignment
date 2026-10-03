variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "ap-southeast-2"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "enable_gpu" {
  description = "Set to true to deploy the optional GPU + vLLM LLM inference node instead of the default CPU + LightGBM node"
  type        = bool
  default     = false
}

variable "cpu_instance_type" {
  description = "Instance type for the default CPU (LightGBM) compute node"
  type        = string
  default     = "t3.medium"
}

variable "gpu_instance_type" {
  description = "Instance type for the optional GPU (vLLM) compute node"
  type        = string
  default     = "g4dn.xlarge"
}
variable "ubuntu_ami_id" {
  description = "Ubuntu 22.04 AMI ID in us-east-1"
  type        = string
  default     = "ami-0e1bed4f06a3b463d" # Ubuntu 22.04 LTS us-east-1 (example)
}

variable "gpu_ami_id" {
  description = "AMI for optional GPU path"
  type        = string
  default     = ""
}