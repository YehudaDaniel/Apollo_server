let express = require('express');
let formidable = require('formidable');
let path = require('path');
let fs = require('fs');
let { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Telling express to parse the incoming data into an object called request.body
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    console.log("works!");
    res.sendStatus(200);
});

app.post('/sendFileToModel', (req, res) => {
    const form = new formidable.IncomingForm();
    
    form.parse(req, (err, fields, files) => {
        if (err) {
            console.error(err);
            return res.sendStatus(500);
        }

        // Accessing the first file in the array
        const file = files.file[0]; 

        if (!file) {
            return res.status(400).send('No file was uploaded.');
        }

        const oldPath = file.filepath; 
        const newPath = path.join(__dirname, 'uploads', file.originalFilename);

        // Move the file to the desired location
        fs.rename(oldPath, newPath, function (err) {
            if (err) {
                console.error(err);
                return res.sendStatus(500);
            }
            
            console.log(`File saved to ${newPath}`);
            
            // Run the Python script to process the audio file
            const pythonProcess = spawn('python', ['src/process_audio.py', newPath]);

            pythonProcess.stdout.on('data', (data) => {
                console.log(`stdout: ${data}`);
            });

            pythonProcess.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
            });

            pythonProcess.on('close', (code) => {
                console.log(`Python script exited with code ${code}`);
                res.status(200).send(`File received and processed. Python script exited with code ${code}`);
            });
        });
    });
});

const server = app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
