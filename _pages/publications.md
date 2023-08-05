---
layout: page
permalink: /publications/
title: publications
years: [2022, 2021, 2018]
nav: true
---

<div class="publications">

{% for y in page.years %}
{% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>