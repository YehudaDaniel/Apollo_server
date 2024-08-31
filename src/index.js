let express = require('express');
let formidable = require('formidable');

const app = express();
const PORT = process.env.PORT || 3000;

//Telling express to parse the incoming data into an object called request.body
app.use(express.json());
app.use(express.urlencoded({ extended:true }));

app.get('/', (req, res) => {
    console.log("works!");
    res.sendStatus(200);
});

app.post('/sendFileToModel', (req, res) => {
    const form = new formidable.IncomingForm();
    form.parse(req, (err, fields, files) => {
        if(err){
            res.sendStatus(500);
        }
        const file = files[0];
        console.log(file);
    });
    console.log(req.files);
    
    res.sendStatus(200);
});

const server = app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});