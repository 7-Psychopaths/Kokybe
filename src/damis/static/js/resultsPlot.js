(function() {
	window.resultsPlot = {
		generate: function(offset, amplitude) {
			var res = [];
			var start = 0,
			end = 10;

			for (var i = 0; i <= 50; ++i) {
				var x = start + i / 50 * (end - start);
				res.push([x, amplitude * Math.sin(x + offset)]);
			}

			return res;
		},

		// custom color palette, rotates through a range of hue values
		generateColorPalette: function(data) {
			var len = data.length;
			return $.map(data, function(o, i) {
				return jQuery.Color({
					hue: (i * 360 / len),
					saturation: 0.95,
					lightness: 0.35,
					alpha: 1
				}).toHexString();
			});
		},

		generateSymbolPalette: function(data) {
			var baseOptions = [ ["circle", gettext("Circle")], ["square", gettext("Square")], ["diamond", gettext("Diamond")], ["triangle", gettext("Triangle")], ["cross", gettext("Cross")] ];
            var allOptions= [];
            for (var i = 0; i < data.length; i++) {
                allOptions.push(baseOptions[i % baseOptions.length]);
            }
            return allOptions;
		},

		renderChart: function(plotContainer, plotPlaceholder, data, colors, symbols) {
			var data = [];
			$.each(initData, function(idx, rec) {
				data.push({
					label: rec['group'],
					points: {
						symbol: symbols[idx][0],
					},
					data: rec['data'],
					color: colors[idx],
				});
			});

			options = {
				series: {
					points: {
						show: true,
						radius: 3
					},
				},
				grid: {
					clickable: true,
					hoverable: true
				}
			};

			var plot = $.plot(plotPlaceholder, data, options);

			$(plotPlaceholder).bind("plotclick", function(event, pos, item) {
				if (item) {
					$("<div id='point-tooltip'></div>").css({
						position: "absolute",
						display: "none",
						border: "1px solid #fdd",
						padding: "2px",
						"background-color": "#fee",
						opacity: 0.80
					}).appendTo(plotContainer);

					var x = item.datapoint[0].toFixed(2),
					y = item.datapoint[1].toFixed(2);

					var containerOffset = $(plotPlaceholder).offset();
					$("#point-tooltip").html(x + ", " + y + " (" + gettext("index") + ": " + item.dataIndex + ", " + gettext("class") + ": " + item.series.label + ")").css({
						top: item.pageY - containerOffset['top'],
						left: item.pageX - containerOffset['left'] + 10
					}).fadeIn(200);
				} else {
					$("#point-tooltip").hide();
				}
			});
		},

		// TODO: sends Ajax request to obtain results for the connected component
		getDataToRender: function() {
			initData = [{
				group: "1",
				data: window.resultsPlot.generate(2, 1.8)
			},
			{
				group: "2",
				data: window.resultsPlot.generate(4, 0.9)
			},
			{
				group: "3",
				data: window.resultsPlot.generate(7, 1.1)
			},
			{
				group: "4",
				data: window.resultsPlot.generate(10, 0.2)
			}];
			return initData;
		},

		chart: function(formWindow) {
			var plotContainer = formWindow.find(".plot-container");
			if (plotContainer.length == 0) {
				var plotContainer = $("<div class=\"plot-container\" style=\"min-height: 400px; position: relative;\"></div>");
				var plotPlaceholder = "<div class=\"results-container\" style=\"width: 600px; height: 300px; margin: auto;\"></div>";
				plotContainer.append(plotPlaceholder);
				plotContainer.append("<table id=\"fieldsTable\" style=\"margin: auto; margin-top: 20px;\"><thead><th>" + gettext('Class') + "</th><th>" + gettext('Color') + "</th><th>" + gettext('Shape') + "</th></thead><tbody></tbody></table>");
				formWindow.append(plotContainer);

				var initData = window.resultsPlot.getDataToRender();
				var colorPalette = window.resultsPlot.generateColorPalette(initData);
				var symbolPalette = window.resultsPlot.generateSymbolPalette(initData);

				var fieldsTableBody = $("#fieldsTable tbody");
				$.each(initData, function(idx, series) {
					var seriesRow = $("<tr></tr>");
					seriesRow.append("<td>" + idx + "</td>");
					seriesRow.append("<td><input type=\"text\" value=\"" + colorPalette[idx].toLowerCase() + "\"/></td>");

					var shapeSelect = $("<select></select>");
					$.each(symbolPalette, function(i, shape) {
						shapeSelect.append("<option value=\"" + shape[0] + "\"" + (idx == i ? "selected=\"selected\"": "") + ">" + shape[1] + "</option>");
					});
					var shapeCell = $("<td></td>");
					shapeCell.append(shapeSelect);
					seriesRow.append(shapeCell);

					fieldsTableBody.append(seriesRow);
				});

				window.resultsPlot.renderChart(plotContainer, ".results-container", initData, colorPalette, symbolPalette);

				// customize dialog
				formWindow.dialog("option", "close", function() {
					$(this).find("#point-tooltip").remove();
				});
				var buttons = formWindow.dialog("option", "buttons");
				buttons.splice(0, 0, {
					text: gettext('Download'),
					click: function(ev) {}
				});
				buttons.splice(0, 0, {
					text: gettext('Update'),
					click: function(ev) {}
				});
				formWindow.dialog("option", "buttons", buttons);
				formWindow.dialog("option", "minWidth", 650);
			}
		},
	}
})();

