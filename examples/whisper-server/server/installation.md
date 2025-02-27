
# Detailed Installation Instructions

This guide explains the new project structure, where initialization is split into three scripts: 
- **model_init.sh** – downloads, compiles, and installs the Whisper model (and builds whisper.cpp), 
- **server_init.sh** – installs backend dependencies and starts the server, 
- ~~**frontend_init.sh** – (optional) installs dependencies and starts the frontend.~~

---

## 1. Install Required System Packages

### Linux (Ubuntu/Debian)
```bash
# Basic tools
sudo apt update
sudo apt install -y build-essential cmake git ffmpeg curl

# Install NVM (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
source ~/.bashrc

# Verify nvm is installed:
nvm --version

# Install and use Node.js using brew or with nvm:
brew install node

# Install pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -
source ~/.bashrc
```

### macOS
```bash
# Install Homebrew if not already installed:
 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Basic tools
brew install cmake git ffmpeg


# Install NVM (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
source ~/.bashrc

# Install and use Node.js with brew:
brew install node
```


~~```bash~~
~~nvm use 18~~   `# no longer required after switching from nodejs-whisper`
~~nvm use~~   `# to compiled binary (whisper.cpp with ggml model)`

Install pnpm

```bash
# Install pnpm
brew install pnpm
```

### Windows
1. Install [Git for Windows](https://gitforwindows.org/).
2. Install [NVM for Windows](https://github.com/coreybutler/nvm-windows/releases) (follow the installer instructions).
3. Open PowerShell or CMD as administrator:
   ```powershell
   nvm install
   ```
4. Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
5. Install [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) and ensure it’s in your PATH.
6. Install pnpm:
   ```powershell
   iwr https://get.pnpm.io/install.ps1 -useb | iex
   refreshenv
   ```

---

## 3. Model Initialization: `model_init.sh`

1. Navigate to the `server` directory:
   ```bash
   cd server
   ```
2. Run the script:
   ```bash
    chmod +x model_init.sh
    ./model_init.sh
                        # or simply
    bash model_init.sh
   ```
   
   The script will:
   - Check for required tools (node, pnpm, curl, git, cmake),
   - Create directories: `uploads`, `models`, and `build`,
   - Allow you to select and download a Whisper model (with size verification),
   - Optionally clone, build, and install whisper.cpp (when prompted: “Do you want to compile whisper.cpp?”).

   After completion:
   - The `models` directory will contain the selected model (e.g., `ggml-large-v3-turbo.bin`) and the compiled binary (e.g., `whisper-cli`),
   - The `build` directory will have the compilation files.

---

## 4. Server Initialization: `server_init.sh`

1. In the `server` directory, run:
   ```bash
   ./server_init.sh
   ```
   This script will:
   - Check for pnpm,
   - Install backend dependencies (`pnpm install`),
   - Start the development server on port `3001` (`pnpm dev`).

2. **Test the Server:**
   Open a new terminal and run:
   ```bash
   curl http://localhost:3001/health
   ```
   A response (e.g., “OK”) confirms that the backend is running properly.

---

## Developer Configuration

- **Backend (port 3001):**
  - Accessible at [http://localhost:3001](http://localhost:3001)
  - Accepts requests from the frontend (e.g., [http://localhost:3000](http://localhost:3000)).

---

## Troubleshooting

### ~~"Node.js version must be 18.x"
~```bash
node --version
nvm install 18
nvm use
~~```
~~Remember to run `nvm use` in every new terminal session.~~ 
    > `# nodejs 18 no longer required`
### "pnpm not found"
```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
source ~/.bashrc
```

### "ffmpeg not found"
```bash
ffmpeg -version
```
*If FFmpeg is not recognized, install it (e.g., `sudo apt install ffmpeg` on Ubuntu).*

### "Port 3000/3001 is already in use"
```bash
# Linux/macOS
lsof -i :3000
kill -9 <PID>

# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## Post-Installation Testing

1. **Test the Server:**
   ```bash
   cd server
   pnpm dev
   # In a new terminal:
   curl http://localhost:3001/health
   ```

(...) you can then project your own frontend application
---

This guide provides a clear and streamlined installation and configuration process using nvm for Node.js 18, with dedicated scripts for the model, server, and frontend initialization.

### MIT License

### Copyright (c) 2025 Maciej Gad | contact at `red` and `yellow` and `green` on **[div0.space]**




