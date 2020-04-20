 var firebaseConfig = {
  apiKey: "##your api key###",
  authDomain: "##your auth domain##",
  databaseURL: "##your database url##",
  projectId: "##your project id ##",
  storageBucket: "##your storage bucket##",
  messagingSenderId: "##your messaging sender id##",
  appId: "#your app id##"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);
console.log(firebase);

function uploadImage() {
  const ref = firebase.storage().ref();
  const file = document.querySelector("#photo").files[0];
  const name = +new Date() + "-" + file.name;
  const metadata = {
	contentType: file.type
  };
  const task = ref.child(name).put(file, metadata);
  task
	.then(snapshot => snapshot.ref.getDownloadURL())
	.then(url => {
	  console.log(url);
	  document.querySelector("#image").src = url;
	})
	.catch(console.error);
}
