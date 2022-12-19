// JavaScript library for visualizing few-shot learning task and model responses

var colors = ['#ff0000', '#0000ff', '#33cc33', '#b7b600', '#ce9fcf', '#00b0b3']; // red, blue, green, yellow

// var input_dict_manual = {};
// input_dict_manual['1'] = 'red';
// input_dict_manual['2'] = 'blue';
// input_dict_manual['3'] = 'green';
// input_dict_manual['DAX'] = 'yellow';
// input_dict_manual['after'] = 'after';
// input_dict_manual['surround'] = 'surround';
// input_dict_manual['thrice'] = 'thrice';

var input_dict_manual = {};
input_dict_manual['1'] = 'dax';
input_dict_manual['2'] = 'lug';
input_dict_manual['3'] = 'wif';
input_dict_manual['DAX'] = 'zup';
input_dict_manual['after'] = 'kiki';
input_dict_manual['surround'] = 'blicket';
input_dict_manual['thrice'] = 'fep';

var output_dict_manual = {};
output_dict_manual['1'] = colors[0];
output_dict_manual['2'] = colors[1];
output_dict_manual['3'] = colors[2];
output_dict_manual['DAX'] = colors[3];

var stims_support = [
	['1','1'],
	['3','3'],
	['2','2'],
	['DAX','DAX'],
	['2 after 3','3 2'],
	['1 after 2','2 1'],
	['2 thrice','2 2 2'],
	['2 surround 3','2 3 2'],
	['1 thrice','1 1 1'],
	['3 surround 1','3 1 3'],
	['2 thrice after 3','3 2 2 2'],
	['3 after 1 surround 2','1 2 1 3'],
	['2 after 3 thrice','3 3 3 2'],
	['3 surround 1 after 2','2 3 1 3'],
];		

var convert_command_to_words = function (mycommand) {
	// convert an abstract command to pseudoword sequence
	mycommand = mycommand.split(" ");
	var mywords = [];
	for (var i=0; i<mycommand.length; i++) {
		mywords.push( input_dict_manual[mycommand[i]] );
	}
	return mywords.join(" ");
};

var make_circle = function (mycolor, myclass) {
	var mycircle = $("<li>").attr('class',myclass).html(
					$("<span>").attr('style',"color:"+mycolor+';').html("&#x25CF")
				);
	return mycircle;
};

var make_circles_cell = function (mystimulus) {		
	var mylist = $("<ul>");
	if (mystimulus.length > 0) {
		var mysequence = mystimulus.split(" ");
		for (var i=0; i<mysequence.length; i++) {
			var mycircle = make_circle( output_dict_manual[mysequence[i]], 'data_li');
			$(mylist).append(mycircle);
		}
	}
	return $("<td>").append(mylist);
};

var make_example_table = function (mystimuli, ncol) {
	// make table that shows command and outputs					
	var space_symbol = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;';
	var nrow = Math.ceil(mystimuli.length / ncol);
	var sc = 0;
	var mytable = $('<table>');
	for (var r=0; r<nrow; r++) {
		var myrow = $("<tr>");
		for (var c=0; c<ncol; c++) {
			var space_cell = $("<td>").html(space_symbol);
			var count_cell = '';
			if (sc < mystimuli.length) {
				var command_cell = $("<td>").html(convert_command_to_words(mystimuli[sc][0]));
				var circle_cell = make_circles_cell(mystimuli[sc][1]);								
				if (mystimuli[sc].length > 2) {
					count_cell = $("<td>").text('('+mystimuli[sc][2]+')');
				}
			}
			else {
				var command_cell = $("<td>").html('');
				var circle_cell = $("<td>").html('');
			}				
			$(myrow).append( command_cell );
			$(myrow).append( circle_cell );
			$(myrow).append( count_cell );
			$(myrow).append( space_cell );
			sc += 1;
		}
		mytable.append(myrow);
	}
	return mytable;
};

var main = function(){
	$('#title').text(title);
	$('#support').html(make_example_table(stims_support,2));
	ntrial = all_data.length;
	for (var i=0; i<ntrial; i++) {
		mytrial = all_data[i]
		$('#queries').append(make_example_table(mytrial,2));
		$('#queries').append('<hr>')
	}
};