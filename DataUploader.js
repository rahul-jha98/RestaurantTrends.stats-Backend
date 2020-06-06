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

var upload = async (localFile, remoteFile, type) => {

  let uuid = UUID();
  var content = '';
  if (type == 'image') {
      content = 'image/png';
  } else {
      content = 'text/html';
  }
  return bucket.upload(localFile, {
        destination: remoteFile,
        uploadType: "media",
        metadata: {
          contentType: content,
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
    await upload(city_name + '/' + path, city_name + '/' + path).then(url => {
        maindata[key].url = url;
    })
    
}


async function manageAll(allKeys) {
    var i = 0;
    var total = allKeys.length;
    for(const key of allKeys) {
        await handleUpload(key);
        i = i + 1;
        console.log("Uploded " + i + " of " + total);
    ;};
    
    axios.put(databaseURL + city_name + ".json", jsondata).then(res => {
        console.log('\n\nAll the data has been uploaded to firebase. ')
        console.log(city_name + " can now be viewed in website")
    }).catch(error => {
        console.log(error)
    }) 
}

manageAll(allkeys)

