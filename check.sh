#!/bin/sh
if [ -z "$1" ]; then
  echo 'Enter filename...Orz'
  exit
fi

cat $1|wc -l
ls data/$1 |wc -l

