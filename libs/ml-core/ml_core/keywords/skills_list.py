"""Comprehensive list of technical skills and keywords."""

# Programming Languages
PROGRAMMING_LANGUAGES = {
  "python",
  "java",
  "javascript",
  "typescript",
  "go",
  "golang",
  "rust",
  "c++",
  "cpp",
  "c#",
  "csharp",
  "ruby",
  "php",
  "swift",
  "kotlin",
  "scala",
  "r",
  "perl",
  "shell",
  "bash",
  "powershell",
  "sql",
  "html",
  "css",
  "dart",
  "elixir",
  "haskell",
  "lua",
  "objective-c",
}

# Frameworks & Libraries
FRAMEWORKS = {
  # Python
  "django",
  "flask",
  "fastapi",
  "pytest",
  "numpy",
  "pandas",
  "pytorch",
  "tensorflow",
  "scikit-learn",
  "sklearn",
  "keras",
  "asyncio",
  "celery",
  # JavaScript/TypeScript
  "react",
  "vue",
  "angular",
  "nextjs",
  "express",
  "nodejs",
  "nest",
  "svelte",
  "jquery",
  "webpack",
  "vite",
  "redux",
  "mobx",
  # Java
  "spring",
  "springboot",
  "hibernate",
  "junit",
  "maven",
  "gradle",
  # .NET
  "dotnet",
  "asp.net",
  "blazor",
  "entity-framework",
  # Other
  "rails",
  "laravel",
  "symfony",
  "gin",
  "echo",
  "fiber",
}

# Databases
DATABASES = {
  # Relational
  "postgresql",
  "postgres",
  "mysql",
  "mariadb",
  "sqlite",
  "oracle",
  "mssql",
  "sql-server",
  "db2",
  # NoSQL
  "mongodb",
  "redis",
  "cassandra",
  "dynamodb",
  "couchdb",
  "neo4j",
  "elasticsearch",
  "opensearch",
  # Data Warehouses
  "snowflake",
  "bigquery",
  "redshift",
  "databricks",
}

# Cloud Platforms & Services
CLOUD_PLATFORMS = {
  # AWS
  "aws",
  "ec2",
  "s3",
  "lambda",
  "dynamodb",
  "rds",
  "cloudformation",
  "cloudwatch",
  "sns",
  "sqs",
  "api-gateway",
  "eks",
  "ecs",
  "fargate",
  "cloudfront",
  "route53",
  "vpc",
  "iam",
  "cognito",
  "amplify",
  # GCP
  "gcp",
  "google-cloud",
  "compute-engine",
  "cloud-storage",
  "cloud-run",
  "cloud-functions",
  "gke",
  "bigquery",
  "firestore",
  "pub-sub",
  # Azure
  "azure",
  "azure-functions",
  "azure-storage",
  "azure-sql",
  "aks",
  "azure-devops",
  "cosmos-db",
  # Other Cloud
  "digitalocean",
  "linode",
  "heroku",
  "vercel",
  "netlify",
  "cloudflare",
}

# DevOps & Infrastructure
DEVOPS_TOOLS = {
  "docker",
  "kubernetes",
  "k8s",
  "terraform",
  "ansible",
  "jenkins",
  "gitlab-ci",
  "github-actions",
  "circleci",
  "travis-ci",
  "argocd",
  "helm",
  "istio",
  "prometheus",
  "grafana",
  "datadog",
  "new-relic",
  "puppet",
  "chef",
  "vagrant",
  "packer",
  "vault",
  "consul",
}

# API & Protocols
API_PROTOCOLS = {
  "rest",
  "restful",
  "graphql",
  "grpc",
  "soap",
  "websocket",
  "http",
  "https",
  "tcp",
  "udp",
  "mqtt",
  "amqp",
  "kafka",
}

# Message Queues & Streaming
MESSAGE_SYSTEMS = {
  "kafka",
  "rabbitmq",
  "activemq",
  "redis-streams",
  "nats",
  "pulsar",
  "kinesis",
  "eventbridge",
}

# Version Control & Collaboration
VERSION_CONTROL = {
  "git",
  "github",
  "gitlab",
  "bitbucket",
  "svn",
  "mercurial",
}

# Testing & Quality
TESTING_TOOLS = {
  "jest",
  "mocha",
  "chai",
  "cypress",
  "selenium",
  "playwright",
  "pytest",
  "unittest",
  "testng",
  "junit",
  "postman",
  "insomnia",
}

# Security
SECURITY_TOOLS = {
  "oauth",
  "jwt",
  "saml",
  "ldap",
  "ssl",
  "tls",
  "https",
  "penetration-testing",
  "owasp",
  "security-audit",
  "encryption",
}

# Operating Systems
OPERATING_SYSTEMS = {
  "linux",
  "unix",
  "ubuntu",
  "centos",
  "debian",
  "rhel",
  "fedora",
  "windows",
  "macos",
  "alpine",
}

# Monitoring & Logging
MONITORING_TOOLS = {
  "prometheus",
  "grafana",
  "elk",
  "elasticsearch",
  "logstash",
  "kibana",
  "splunk",
  "datadog",
  "new-relic",
  "sentry",
  "cloudwatch",
}

# Build Tools & Package Managers
BUILD_TOOLS = {
  "npm",
  "yarn",
  "pnpm",
  "pip",
  "poetry",
  "conda",
  "maven",
  "gradle",
  "make",
  "cmake",
  "cargo",
  "composer",
  "bundler",
  "nuget",
}

# Frontend Technologies
FRONTEND = {
  "html",
  "css",
  "sass",
  "scss",
  "less",
  "tailwind",
  "bootstrap",
  "material-ui",
  "chakra-ui",
  "styled-components",
  "svg",
  "canvas",
}

# Mobile Development
MOBILE = {
  "ios",
  "android",
  "react-native",
  "flutter",
  "xamarin",
  "ionic",
  "swift",
  "kotlin",
  "objective-c",
}

# Data & ML
DATA_ML = {
  "machine-learning",
  "ml",
  "deep-learning",
  "ai",
  "nlp",
  "computer-vision",
  "data-science",
  "data-engineering",
  "etl",
  "data-pipeline",
  "mlops",
  "model-training",
  "feature-engineering",
  "neural-networks",
}

# Combine all skills
ALL_SKILLS = (
  PROGRAMMING_LANGUAGES
  | FRAMEWORKS
  | DATABASES
  | CLOUD_PLATFORMS
  | DEVOPS_TOOLS
  | API_PROTOCOLS
  | MESSAGE_SYSTEMS
  | VERSION_CONTROL
  | TESTING_TOOLS
  | SECURITY_TOOLS
  | OPERATING_SYSTEMS
  | MONITORING_TOOLS
  | BUILD_TOOLS
  | FRONTEND
  | MOBILE
  | DATA_ML
)

# Common variations and aliases
SKILL_ALIASES = {
  "k8s": "kubernetes",
  "js": "javascript",
  "ts": "typescript",
  "pg": "postgresql",
  "mongo": "mongodb",
  "tf": "tensorflow",
  "sklearn": "scikit-learn",
  "node": "nodejs",
  "gcp": "google-cloud",
  "aws-lambda": "lambda",
  "aws-s3": "s3",
}
