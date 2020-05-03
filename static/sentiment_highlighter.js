var array = [];
var negative = [];
$.getJSON('https://raw.githubusercontent.com/umadjoe440/Data_Final/master/sentiment_words.json', function (json) {
  for (var key in json) {
      if (json.hasOwnProperty(key)) {
          var item = json[key];
          array.push({
              term: item.term,
              value: +item.value,
              
          });            
      }
  }
console.log(array[3].value)


  for (var i = 0; i < array.length; i++)
  {
      if (array[i].value < 0)
      {
          negative.push(array[i]);
      }
  }
    console.log(negative)
  });
var textbox = document.getElementsByTagName('p')[0];

var highlightSWords = function() {
  //foreach word, perform highlighting actions
  negative.forEach(function(word) {

    //build regular expression to match against each variable (word) in our list
    var regex = new RegExp("(" + " " + word.term + " " + ")", "gi");
    //g = can find multiple instances (global)
    //i = case insenstive

    //replace predefined textbox HTML with 'highlighted' version
    //the regex will find matching words and wrap them in <strong> tags
    //the $1 represents the matched word
    textbox.innerHTML = textbox.innerHTML.replace(regex, "<mark>$1</mark>");
  });


}
