const fs = require('fs');
const THREE = require('node_modules/three');

// Load STL file
const loader = new THREE.STLLoader();
loader.load('AirplaneAllFiles\AirplaneForFreestl.stl', function (geometry) {
    // Convert to JSON
    const json = geometry.toJSON();

    // Save JSON to file
    fs.writeFileSync('AirplaneAllFiles\output.json', JSON.stringify(json));
});
