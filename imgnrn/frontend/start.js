const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Check and install dependencies
function checkDependencies() {
    console.log('Checking dependencies...');

    const packages = [
        'express',
        'multer',
        'cors',
        'onnxruntime-node'
    ];

    packages.forEach(pkg => {
        try {
            require.resolve(pkg);
            console.log(`✓ ${pkg} is installed`);
        } catch (e) {
            console.log(`Installing ${pkg}...`);
            execSync(`npm install ${pkg}`, { stdio: 'inherit' });
        }
    });
}

// Start server
function startServer() {
    checkDependencies();

    const app = require('./server.js');
    console.log('Server started successfully!');
}

// Export for testing
module.exports = { startServer, checkDependencies };

if (require.main === module) {
    startServer();
}
