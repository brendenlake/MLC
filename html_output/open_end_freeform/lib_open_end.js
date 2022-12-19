// JavaScript library for visualizing few-shot learning task and model responses
var output_dict_manual = {};
 output_dict_manual['RED']='#ff0000';
 output_dict_manual['BLUE']='#0000ff';
 output_dict_manual['GREEN']='#33cc33';
 output_dict_manual['YELLOW']='#b7b600';
 output_dict_manual['PINK']='#fb00ff';
 output_dict_manual['PURPLE']='#9d00ff';
 output_dict_manual['color:#ff0000;']='#ff0000';
 output_dict_manual['color:#0000ff;']='#0000ff';
 output_dict_manual['color:#33cc33;']='#33cc33';
 output_dict_manual['color:#b7b600;']='#b7b600';
 output_dict_manual['color:#fb00ff;']='#fb00ff';
 output_dict_manual['color:#9d00ff;']='#9d00ff';
 output_dict_manual['color:#00b0b3;']='#00b0b3';
 output_dict_manual['color:#ce9fcf;']='#ce9fcf';

var convert_command_to_words = function (mycommand) {
	// convert an abstract command to pseudoword sequence
	return mycommand;
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
	var count = 0;
	for (var i=0; i<all_data.length; i+=2) {
		sdata1 = all_data[i]
		sdata2 = all_data[i+1]
		$('#data').append('Episode ' + (count) + '<br><br>');						        
		count++;
        $('#data').append(sdata1[0]);
		$('#data').append(make_example_table(sdata1.slice(1),1));
		$('#data').append('<br>');
        $('#data').append(sdata2[0]);
        $('#data').append(make_example_table(sdata2.slice(1),1));
        $('#data').append('<hr>');
	}
};