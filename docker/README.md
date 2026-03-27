# Docker Configuration for Email Classifier

## 🐳 **What is Docker?**

Docker is a platform that packages applications and their dependencies into **containers** - lightweight, portable environments that run consistently anywhere.

### **Why Use Docker for Your Project?**
- **Consistent Environment**: Same setup on Windows, Mac, Linux
- **Dependency Management**: All Python packages pre-installed
- **Easy Deployment**: One command to run your entire application
- **Isolation**: No conflicts with your local Python installation

---

## 🏗️ **Docker Architecture Explained**

### **Container vs Virtual Machine:**
```
Virtual Machine:     Container:
┌─────────────────┐   ┌─────────────────┐
│   Host OS       │   │   Host OS       │
│ ┌─────────────┐ │   │ ┌─────────────┐ │
│ │ Guest OS    │ │   │ │ Docker App  │ │
│ │ ┌─────────┐ │ │   │ │ + Libraries│ │
│ │ │ Python  │ │ │   │ │ + Dependencies│ │
│ │ │ App     │ │ │   │ └─────────────┘ │
│ │ └─────────┘ │ │   └─────────────────┘
│ └─────────────┘ │
└─────────────────┘
```

### **Email Classifier in Docker:**
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                        │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Python    │  │  Pandas     │  │ Scikit-Learn│      │
│  │   3.10      │  │  1.5.0      │  │   1.1.0     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │    XGBoost  │  │   NLTK      │  │  Transformers│      │
│  │   1.6.0     │  │   3.6.0     │  │   4.20.0    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │               Email Classifier Code             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │   Models    │  │  Strategies │  │ Preprocessing│ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 **How to Use Docker**

### **Prerequisites:**
1. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Restart your computer after installation

### **Quick Start:**
```bash
# Navigate to project root
cd C:\python\email_classifier

# Build and run the container
docker-compose -f docker/docker-compose.yml up --build
```

### **What Happens When You Run This Command?**

#### **Step 1: Build Phase**
```bash
docker-compose up --build
```
**What Docker does:**
1. **Reads Dockerfile** - Instructions for building your container
2. **Downloads Python 3.10 image** - Base operating system with Python
3. **Installs dependencies** - All packages from requirements.txt
4. **Copies your code** - Your src/, scripts/, tests/ folders
5. **Creates container image** - Ready-to-run snapshot

#### **Step 2: Run Phase**
**What Docker does:**
1. **Starts container** - Like a mini-computer with your app
2. **Mounts volumes** - Links your local folders to container
3. **Sets environment** - PYTHONPATH and other variables
4. **Runs your command** - Executes `python scripts/simple_test.py`

---

## 📁 **Docker Files Explained**

### **Dockerfile** - Container Recipe
```dockerfile
# Step 1: Choose base image
FROM python:3.10-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y gcc g++

# Step 4: Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Step 5: Copy your code
COPY src/ ./src/

# Step 6: Set default command
CMD ["python", "scripts/simple_test.py"]
```

### **docker-compose.yml** - Multi-Container Orchestration
```yaml
services:
  email-classifier:
    build: .                    # Build from Dockerfile
    volumes:                    # Link local folders
      - ../data:/app/data
    ports:                      # Expose port
      - "8000:8000"
    environment:                # Set variables
      - PYTHONPATH=/app/src
```

---

## 🎯 **Practical Usage Examples**

### **1. Run Quick Test:**
```bash
docker-compose up --build
# Output: Running your email classifier with test data
```

### **2. Run Specific Script:**
```bash
docker-compose run email-classifier python scripts/run_strategies.py
```

### **3. Start Jupyter for Development:**
```bash
docker-compose up jupyter
# Then open http://localhost:8888 in your browser
```

### **4. Access Container Shell:**
```bash
docker-compose exec email-classifier bash
# Now you're inside the container!
```

---

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: "Command not found: docker"**
**Solution:** Install Docker Desktop from docker.com

### **Issue 2: "Port already in use"**
**Solution:** Change port in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Use different port
```

### **Issue 3: "Permission denied"**
**Solution:** Run Docker Desktop as Administrator

### **Issue 4: "Build failed"**
**Solution:** Check requirements.txt for correct package names

---

## 🎓 **Learning Benefits**

### **For Your Assignment:**
- **Professional deployment** - Shows industry practices
- **Environment consistency** - No "works on my machine" issues
- **Containerization knowledge** - Valuable skill for future projects

### **Understanding the Process:**
1. **Containerization** - Packaging applications with dependencies
2. **Orchestration** - Managing multiple containers
3. **Volume mounting** - Linking local files to container
4. **Port mapping** - Exposing services to host machine

---

## 🚀 **Next Steps**

### **Experiment with Docker:**
1. **Try different commands** in the container
2. **Modify Dockerfile** to add new features
3. **Create custom scripts** for container deployment
4. **Learn docker-compose** for multi-service applications

### **Professional Usage:**
- **CI/CD pipelines** - Automated testing and deployment
- **Cloud deployment** - AWS, Azure, Google Cloud
- **Microservices** - Breaking applications into small containers

---

## 🎯 **Summary**

**Docker makes your email classifier:**
- ✅ **Portable** - Runs anywhere Docker is installed
- ✅ **Consistent** - Same environment every time
- ✅ **Professional** - Industry-standard deployment
- ✅ **Easy to share** - One command to run everything

**You don't need to be a Docker expert** 
