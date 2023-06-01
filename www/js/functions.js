Array.prototype.joinLines = function () {
	var res = this.join("\n")
	
	if (res)
		res += "\n";
	
	return res;
};

String.prototype.splitLines = function () {
	var list = this.split("\n");
	
	if (list.length > 0 && !list[list.length - 1])
		list.pop();
	
	return list;
};

// Converts a number to a string with a specified number of digits before and after the dot together.
Number.prototype.toPrecision2 = function (digits) {
	var parts = this.toString().split(".");
	var p1 = parts[0], p2 = parts[1] || "", pp = p1 + p2;
	var res = p1;
	
	while (pp.slice(0, 1) == "0") {
		digits += 1;
		pp = pp.slice(1);
	}
	
	if (p1.length < digits) {
		res += "." + p2;
		digits += 1; // Because of the dot.
	}
	
	res = res.slice(0, digits);
	digits = Math.max(digits, p1.length)
	
	while (res.length < digits)
		res += "0";
	
	return res;
};

Object.size = function(obj) {
    var size = 0, key;
    for (key in obj) {
        if (obj.hasOwnProperty(key)) size++;
    }
    return size;
};


function getSearchArgs() {
	var parts = document.location.search.slice(1).split("&");
	var result = { };
	
	$.each(parts, function () {
		var pair = this.split("=");
		
		if (pair.length == 1)
			result[pair[0]] = true;
		else
			result[pair[0]] = pair.slice(1).join("=");
	})
	
	return result;
}

var g_lang;
function setLanguage(lang) {
	g_lang=lang;
	$("*[lang]").hide();
	$("*[lang|=" + lang + "]").show();
	$("#lang-select a").removeClass("active");
	$("#lang-select *[id|=" + lang + "]").addClass("active");
	
	//buttons
	$("input").filter("[type=button]").each(function () {
		if(this.getAttribute(g_lang))
			this.value = this.getAttribute(g_lang);
	});
	//(dropdown) lists
	$("option").each(function () {
		if(this.getAttribute(g_lang))
			this.text = this.getAttribute(g_lang);
	});
	//checkboxes
	$("input").filter("[type=checkbox]").each(function () {
		if(this.getAttribute(g_lang))
			this.nextSibling.nodeValue= this.getAttribute(g_lang);
	});
}


function createElement(tag, attrs, contents) {
	var elem = $(document.createElement(tag));
	
	if (attrs)
		$.each(attrs, function (key, value) {
			elem.attr(attrs);
		});
	
	if (contents)
		$.each(contents, function (key, value) {
			elem.append(value);
		});
	
	return elem;
}

function buildControls() {
	var inputs = $(".input:parent")
	var ws = $(document.createTextNode(" "));
	
	inputs.filter("[type=checkbox], [type=radio]").each(function () {
		var name = $(this).attr("name")
		var value = $(this).attr("value")
		var id = name + "--" + $(this).attr("value");
		var label = createElement("label", { "for" : id }, $(this).contents());
		var elem = createElement("input", {
			"name" : name,
			"type" : $(this).attr("type"),
			"value" : $(this).attr("value"),
			"id" : id
		});
		
		$(this).after(label);
		$(this).replaceWith(elem);
		
		label.addClass("label-after");
		elem.click(function () {
			var elem = $(this);
			var type = elem.attr("type");
			var value;
			
			if (type == "checkbox")
				if (elem.length == 1)
					value = (elem.fieldValue().length != 0).toString();
				else
					value = elem.fieldValue().join(" ");
			else if (type == "radio")
				value = elem.fieldValue().join(" ");
			else
				value = elem.fieldValue()[0];
			
			inputValues[elem.attr("name")] = value;
			settingChanged(elem.attr("name"), value);
			
		});
		
		if (inputValues[name] && $.inArray(value, inputValues[name].split(" ")) != -1)
			elem.attr("checked", "checked");
	});
	
	inputs.filter("[type=slider]").each(function () {
		var range = $(this).attr("value").split(" ");
		var value = createElement("span");
		var label = createElement("span", { }, $(this).contents());
		var elem = createElement("div", {
			"class" : $(this).attr("class"),
			"type" : $(this).attr("type"),
			"name" : $(this).attr("name"),
			"id" : $(this).attr("id")
		});
		
		label.addClass("label-before");
		value.addClass("label-after");
		
		$(this).before(label);
		$(this).after(value);
		$(this).replaceWith(createElement("div", {
			"style" : "display: inline-block; vertical-align: middle; position: relative; top: -2px"
		}, elem));
		
		elem.slider({
			"animate" : true,
			"min" : 0,
			"max" : 1,
			"step" : 1 / 1000000,
			"range" : "min",
			"slide" : function (event, ui) {
				var a = parseFloat(range[0]), b = parseFloat(range[1]);
				var x = ui.value;
				
				if (range[2] == "log")
					x = Math.exp((x * (Math.log(b) - Math.log(a)) + Math.log(a)));
				else
					x = x * (b - a) + a;
				
				var val=x.toFixed(0);
				inputValues[$(this).attr("name")] = val;
                                if ($(this).attr("name") == "exposureTime")
                                    $("#exposureTime-ms").text(x.toPrecision2(2)+" ms");
                                else if ($(this).attr("name") == "valueMin")
                                    $("#valueMin-units").text(x.toFixed(0));
                                else if ($(this).attr("name") == "valueMax")
                                    $("#valueMax-units").text(x.toFixed(0));
                                else if ($(this).attr("name") == "saturationMin")
                                    $("#saturationMin-units").text(x.toFixed(0));
                                else
                                    ;
				settingChanged($(this).attr("name"), val);
			}
		});
	});
}

function serializeValues(data) {
	var list = [];
	
	$.each(data, function (key) {
		list.push(key + ": " + this);
	})
	
	return list.joinLines();
}

function parseValues(data) {
	var obj = { };
	
	$.each(data.splitLines("\n"), function () {
		var pos = this.indexOf(":");
		
		obj[$.trim(this.slice(0, pos))] = $.trim(this.slice(pos + 1));
	})
	
	return obj;
}

var outputValueHooks = {
	colorType: function (value) {
		if (value == "gray")
                       // document.getElementById("colorType-debayered").attr("checked", "checked");
                       // document.getElementById("colorType-raw").attr("checked", "unchecked");
			return "8 bit grayscale";
		else if (value == "raw")
			return "8 bit grayscale";
		else if (value == "debayered")
                    //    document.getElementById("colorType-debayered").attr("checked", "unchecked");
                     //   document.getElementById("colorType-raw").attr("checked", "checked");
			return "8 bit RGB";
	},
	imageSensor: function (value) {
		if (value == "Color")
			$("#colorType-section").show();
		else
			$("#colorType-section").hide();
		
		return value;
	},
	perspective: function(value) {
		document.getElementById("perspective-id").selectedIndex=(value*1);
		return(null);
	},
	exposureTime: function(value) {
		var ival=parseInt(value);
		var x=Math.log(ival)/Math.log(1000);
		$("#exposureTime-slider").slider("option", "value", x);
		$("#exposureTime-ms").text(ival.toPrecision2(2)+" ms");
		return(value);
	}, 
        valueMin: function(value) {
		var ival=parseInt(value);
		var x=ival/255;
		$("#valueMin-slider").slider("option", "value", x);
		$("#valueMin-units").text(ival);
		return(value);
	}, 
        valueMax: function(value) {
		var ival=parseInt(value);
		var x=ival/255;
		$("#valueMax-slider").slider("option", "value", x);
		$("#valueMax-units").text(ival);
		return(value);
	},
        saturationMin: function(value) {
		var ival=parseInt(value);
		var x=ival/255;
		$("#saturationMin-slider").slider("option", "value", x);
		$("#saturationMin-units").text(ival);
		return(value);
	},
	autoExposure: function(value) {
		ival=parseInt(value);
		if(ival) {
			$("#exposureTime-slider").slider("disable");
		} else {
			$("#exposureTime-slider").slider("enable");
		}
		document.getElementsByName("autoExposure")[0].checked=ival;
		return(null);
	},
        startButton: function(value) {
	  	ival=parseInt(value);
		if(ival) {			
			$("#startButton-id").text("Stop");
		} else {			
			$("#startButton-id").text("Start");
		}
	}
};


function exchangeState(header, data, onLoad, onError) {
	$.ajax({
		async: true,
		cache: false,
		contentType: "text/plain",
		data: header + "\n" + serializeValues(data),
		error: onError,
		success: function (data) {
			if (onLoad)
				onLoad(parseValues(data));
		},
		timeout: 2000,
		type: "POST",
		url: "/cgi-bin/cgi"
	});
}

var offBanner = {
	active: false,
	show: function () {
		var wheelPos = 0;
		
		if (this.active)
			return;
		
		this.active = true;
		$("#off").stop(true, true).fadeIn("fast");
		
		$("#off .wheel").everyTime("100ms", function () {
			wheelPos = (wheelPos + 1) % 12;
			$(this).css("background-position", -wheelPos * 32 + "px");
		});
	},
	hide: function () {
		if (!this.active)
			return;
		
		this.active = false;
		$("#off").stop(true, true).fadeOut("fast", function () {
			$(".wheel", this).stopTime();
		});
	}
}

var stateControl = {
	currentState: "online",
	pulledState: "",
	pullState: function (state) {
		var that = this;
		
		if (state != this.pulledState) {
			$(document).stopTime("stateControl");
			this.pulledState = state;
		}
		
		if (state != this.currentState) {
			$(document).oneTime("2s", "stateControl", function () {
				that.changeState(state);
			});
		}
	},
	changeState: function (state) {
		var states = {
			offline: function () {
				offBanner.show();
			},
			online: function () {
				offBanner.hide();
			}
		};
		
		this.currentState = state;
		states[state]();
	}
}

var g_changed_values = new Object();

function settingChanged(key, setting) {
	g_changed_values[key]=setting;
}

function autoExposureChanged(key, setting) {
	settingChanged(key, setting);
	
	if(setting) {
		$("#exposureTime-slider").slider("disable");
	} else {
		$("#exposureTime-slider").slider("enable");
	}
}



function startButtonChanged(key, setting) {
        settingChanged(key, setting);

        if(startButtonVal == 1) {			
                $("#startButton-id").text("Stop");
                startButtonVal = 0;
        } else {			
                $("#startButton-id").text("Start");
                startButtonVal = 1;
        }
}

function updateCycle() {
	function getImageInfo() {
		$(document).oneTime("0.1s", function () {
			exchangeState("GetImageInfo", { }, function (data) {
				$.each(data, function (key, value) {
					function id(value) {
						return value;
					};
					var val=(outputValueHooks[key] || id)(value);
					if(val!=null)
						$("#" + key).text(val);
				});
				
				g_changed_values = new Object(); //reset changes
			
				loadImage();
			
				online();
			}, offline);
		});
	}
	
	function getSystemInfo() {
		$(document).oneTime("0.5s", function () {
			exchangeState("GetSystemInfo", { }, function (data) {
				$.each(data, function (key, value) {
					function id(value) {
						return value;
					};
					var val=(outputValueHooks[key] || id)(value);
					if(val!=null)
						$("#" + key).text(val);
				});
				
				getImageInfo();
			
			}, offline);
		});
	}
	
	
	
	function offline() {
		stateControl.pullState("offline");
		getSystemInfo();
	}
	
	function online() {
		stateControl.pullState("online");
	}
	
	offline(); //set to offline at the beginning
	
	
	var tmp_img_val=0;
	
	function loadImage() {
		
		//check for changed values, do not send next image request if there is a setting change
		if(Object.size(g_changed_values) > 0) {
			
			exchangeState("SetOptions", g_changed_values, loadImage, offline);
			g_changed_values=new Object();
			
		} else {
			
			var img = $(new Image());
			
			img.load(function() { 
				//$("#image").replaceWith(this); 
				//this.id="image";
                                online();
				loadImage(); //load next image here instead with timer. prevents overloading the (poor) browser
			});
			
			img.error(offline);
			
			++tmp_img_val;
                        $("#imageWindow1").attr("src", "/cgi-bin/cgi?GetImage_"+tmp_img_val);
			img.attr("src", "/cgi-bin/cgi?GetImage_"+tmp_img_val); //preload the next image                                                				
		}
		
	}
	
}




