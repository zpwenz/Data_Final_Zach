var arr = ['i', 'me', 'myself', 'mine', 'my'];

//locate the element on the page to do the word operations
//this will find the first <p> tag
var textbox = document.getElementsByTagName('p')[0];

var highlightPWords = function() {
  //foreach word, perform highlighting actions
  arr.forEach(function(word) {

    //build regular expression to match against each variable (word) in our list
    var regex = new RegExp("(" + " " + word + " " + ")", "gi");
    //g = can find multiple instances (global)
    //i = case insenstive

    //replace predefined textbox HTML with 'highlighted' version
    //the regex will find matching words and wrap them in <strong> tags
    //the $1 represents the matched word
    textbox.innerHTML = textbox.innerHTML.replace(regex, "<mark>$1</mark>");
    console.log(arr)
  });


}
var arrA = [
  "absolutely",
  "all",
  "always",
  "complete",
  "completely",
  "constant",
  "constantly",
  "definitely",
  "entire",
  "ever",
  "every",
  "everyone",
  "everything",
  "full",
  "must",
  "never",
  "nothing",
  "totally",
  "whole"
]

//locate the element on the page to do the word operations
//this will find the first <p> tag
var textbox = document.getElementsByTagName('p')[0];

var highlightAWords = function() {
  //foreach word, perform highlighting actions
  arrA.forEach(function(word) {

    //build regular expression to match against each variable (word) in our list
    var regex = new RegExp("(" + " " + word + " " + ")", "gi");
    //g = can find multiple instances (global)
    //i = case insenstive

    //replace predefined textbox HTML with 'highlighted' version
    //the regex will find matching words and wrap them in <strong> tags
    //the $1 represents the matched word
    textbox.innerHTML = textbox.innerHTML.replace(regex, "<mark>$1</mark>");
    console.log(arrA)
  });
}
function populateTextarea() {
  //get the text by id or predefined or however you wish or passed to function
  var txt = "All the warnings from the punk rock 101 courses over the years, since my first introduction to the, shall we say, ethics involved with independence and the embracement of your community has proven to be very true. I haven't felt the excitement of listening to as well as creating music along with reading and writing for too many years now. I feel guity beyond words about these things. For example when we're back stage and the lights go out and the manic roar of the crowds begins., it doesn't affect me the way in which it did for Freddie Mercury, who seemed to love, relish in the the love and adoration from the crowd which is something I totally admire and envy. The fact is, I can't fool you, any one of you. It simply isn't fair to you or me. The worst crime I can think of would be to rip people off by faking it and pretending as if I'm having 100% fun. Sometimes I feel as if I should have a punch-in time clock before I walk out on stage. I've tried everything within my power to appreciate it (and I do,God, believe me I do, but it's not enough). I appreciate the fact that I and we have affected and entertained a lot of people. It must be one of those narcissists who only appreciate things when they're gone. I'm too sensitive. I need to be slightly numb in order to regain the enthusiasms I once had as a child. On our last 3 tours, I've had a much better appreciation for all the people I've known personally, and as fans of our music, but I still can't get over the frustration, the guilt and empathy I have for everyone. There's good in all of us and I think I simply love people too much, so much that it makes me feel too fucking sad. The sad little, sensitive, unappreciative, Pisces, Jesus man. Why don't you just enjoy it? I don't know! I have a goddess of a wife who sweats ambition and empathy and a daughter who reminds me too much of what i used to be, full of love and joy, kissing every person she meets because everyone is good and will do her no harm. And that terrifies me to the point to where I can barely function. I can't stand the thought of Frances becoming the miserable, self-destructive, death rocker that I've become. I have it good, very good, and I'm grateful, but since the age of seven, I've become hateful towards all humans in general. Only because it seems so easy for people to get along that have empathy. Only because I love and feel sorry for people too much I guess. Thank you all from the pit of my burning, nauseous stomach for your letters and concern during the past years. I'm too much of an erratic, moody baby! I don't have the passion anymore, and so remember, it's better to burn out than to fade away. Peace, love, empathy. Kurt Cobain Frances and Courtney, I'll be at your alter. Please keep going Courtney, for Frances. For her life, which will be so much happier without me. I LOVE YOU, I LOVE YOU!"

  document.getElementById("text_box").value = txt;
}


// List of words
stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
function remove_stopwords(str) {
  res = []
  words = str.split(' ')
  for(i=0;i<words.length;i++) {
     word_clean = words[i].split(".").join("")
     if(!stopwords.includes(word_clean)) {
         res.push(word_clean)
     }
  }
  return(res)
}  

myWords = document.getElementById("text_box").value
myWords = remove_stopwords(myWords)
var frequency_list = myWords.reduce(function(p, c) {
  p[c] = (p[c] || 0) + 1;
  return p;
}, {});
var array = Object.keys(frequency_list).map(function(key) {
  return { text: key, size: frequency_list[key] };
});
console.log(array)

    
