
window.addEventListener("load", function(e) {
  console.log("Window loaded");
  document.getElementById("ask").addEventListener("click", function(e) {
    console.log("Sending request");
    r = new XMLHttpRequest();
    c = encodeURIComponent(document.getElementById("c").value).replace(/%20/g, '+');
    q = encodeURIComponent(document.getElementById("q").value).replace(/%20/g, '+');
    console.log("Context: "+c);
    console.log("Query: "+q);
    r.open("GET", "http://13.65.198.76:8000/ask?c="+c+"&q="+q);
    r.addEventListener("load", function(e) {
      console.log("Response received");
      document.getElementById("answer").innerHTML = r.responseText;
    });
    r.send();
  });
});