// See an example of usage:
// https://gist.github.com/jherax/8781f45dcd068a9e3e37

//BEGIN MODULE
var jsu = (function (undefined) {
  
    var _toString = Object.prototype.toString,
        //the default parser function
        _parser = function (x) { return x; },
        //gets the item to be sorted
        _getItem = function (x) {
            return this.parser((_toString.call(x) == "[object Object]" && x[this.prop]) || x);
        };

    // Creates a method for sorting the Array
    // @array: the Array of elements
    // @o.prop: property name (if it is an Array of objects)
    // @o.desc: determines whether the sort is descending
    // @o.parser: function to parse the items to expected type
    function _sortBy (array, o) {
        if (_toString.call(array) != "[object Array]" || !array.length)
            return [];
        if (_toString.call(o) != "[object Object]")
            o = {};
        if (_toString.call(o.parser) != "[object Function]")
            o.parser = _parser;
        //if @o.desc is false: set 1, else -1
        o.desc = [1, -1][+!!o.desc];
        return array.sort(function (a, b) {
            a = _getItem.call(o, a);
            b = _getItem.call(o, b);
            return ((a > b) - (b > a)) * o.desc;
        });
    }
    
    //export the public API
    return {
      sortBy: _sortBy
    };
})();
//END MODULE