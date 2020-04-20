const UUID = require("uuid-v4");

const fbId = "zomato-4a45e";
const fbKeyFile = "./service.json";
const {Storage} = require('@google-cloud/storage');
const axios = require('axios')

const fs = require('fs')


const storage = new Storage({
    projectId: fbId,
    keyFilename: fbKeyFile,
  });

const databaseURL=  `https://${fbId}.firebaseio.com/`


const bucket = storage.bucket(`${fbId}.appspot.com`);

var city_name = ""

process.argv.forEach(function (val, index, array) {
   if (index == 2) {
       city_name = val;
   }
});

let rawdata = fs.readFileSync(city_name + '/data.json');
let jsondata = JSON.parse(rawdata);


var upload = async (localFile, remoteFile) => {

  let uuid = UUID();

  return bucket.upload(localFile, {
        destination: remoteFile,
        uploadType: "media",
        metadata: {
          contentType: 'image/png',
          metadata: {
            firebaseStorageDownloadTokens: uuid
          }
        }
      })
      .then((data) => {

          let file = data[0];
          return "https://firebasestorage.googleapis.com/v0/b/" + bucket.name + "/o/" + encodeURIComponent(file.name) + "?alt=media&token=" + uuid;
      });
}


var allkeys = Object.keys(jsondata.data);
maindata = jsondata.data;



async function handleUpload(key) {
    path = maindata[key].path
    await upload(path, path).then(url => {
        maindata[key].url = url;
        console.log(jsondata)
    })
    
}


async function manageAll(allKeys) {
    for(const key of allKeys) {
        await handleUpload(key);
    ;};
    
    axios.put(databaseURL + "Test.json", jsondata).then(res => {
        console.log('Uploaded data')
    }).catch(error => {
        console.log(error)
    }) 
}

manageAll(allkeys)


